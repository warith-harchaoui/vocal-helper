"""Quel diariseur différé gratuit et local est le meilleur, au taux d'erreur seul ?

Le critère
----------
En différé, le facteur temps réel est sans objet : on attend. Le seul critère
est donc la qualité, et elle se lit à quatre endroits — le taux d'erreur et ses
trois composantes, le **nombre de locuteurs** trouvé, et l'accord entre deux
instants éloignés, qui dit si « le même intervenant » veut encore dire quelque
chose une demi-heure plus tard.

Les concurrents
---------------
``pyannote community-1``
    Le défaut actuel. Entraîné sur AMI, entre autres.
``pyannote 3.1``
    La génération précédente, à égalité en médiane et bien pire dans la queue :
    pire réunion 0,474 contre 0,149.
``sherpa``
    Le chemin sans torch, segmentation et empreintes en ONNX.
``nemo sortformer``
    Le modèle de bout en bout de la génération précédente, plafonné à 4
    locuteurs et entraîné sur des extraits de 90 s.
``Nemotron, calendrier hors ligne``
    Le même modèle que le direct, mais avec le calendrier que son export a
    inscrit dans le graphe et vérifié de bout en bout : 30,4 s de tampon au lieu
    de 1,04 s, donc trente fois plus de contexte. Jamais mesuré jusqu'ici.
``Nemotron, calendrier direct``
    Pour voir ce que le contexte apporte, à modèle identique.

Les deux corpus, et pourquoi les deux
-------------------------------------
AMI est réel, ce qui est irremplaçable, mais son annotation est transcrite sur
les **micros-casques** quand on donne au moteur le micro de salle : elle décrit
des mots que le signal ne porte pas, et pyannote l'a dans ses données
d'entraînement. Les mixages synthétiques n'ont aucun de ces biais — vérité
exacte par construction — mais ils sont fabriqués. Aucun des deux ne suffit, et
un moteur qui gagne sur les deux ne doit rien au hasard.

Usage
-----
    python studies/diar_offline_shootout.py --ami <dev-slice> --mix <mix_0000>
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import click
import numpy as np
import soundfile as sf

import vocal_helper as voh
from vocal_helper.consistency import speaker_consistency

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from diar_far_field import reverberate  # noqa: E402
from diar_incremental_vs_greedy import reference_from_rttm  # noqa: E402
from diar_nemotron_threshold import components  # noqa: E402
from rediar_speaker_consistency import reference_from_recipe  # noqa: E402


def contenders() -> dict[str, object]:
    """Les moteurs à comparer, nommés comme ils seront lus.

    Chaque entrée est une fabrique : un moteur qui n'est pas installé doit
    manquer sa ligne, pas faire tomber la campagne.
    """
    return {
        "pyannote community-1": lambda: voh.OfflineDiarStage(backend="pyannote"),
        "sherpa": lambda: voh.OfflineDiarStage(backend="sherpa"),
        "nemo sortformer": lambda: voh.OfflineDiarStage(backend="nemo"),
        "Nemotron hors ligne": lambda: voh.NemotronOnnxDiar(),
        "Nemotron direct": lambda: voh.NemotronOnnxDiar(mode="low_latency"),
    }


def evaluate(
    truth: list[tuple[float, float, str]],
    audio: np.ndarray,
    sr: int,
    label: str,
) -> dict[str, dict]:
    """Faire passer tous les moteurs sur un signal, et rendre leurs chiffres."""
    print(
        f"\n── {label} · {len(audio) / sr / 60:.1f} min · {len({s for _, _, s in truth})} locuteurs"
    )
    print(
        f"   {'moteur':22} {'DER':>6} {'manqué':>7} {'f. alarme':>10} "
        f"{'confusion':>10} {'étiq.':>6} {'≥300s':>7} {'RTF':>6}"
    )
    out: dict[str, dict] = {}
    for name, build in contenders().items():
        try:
            stage = build()
            started = time.perf_counter()
            hypothesis = stage.diarize(audio, sr)
            rtf = (time.perf_counter() - started) / (len(audio) / sr)
        except Exception as problem:  # noqa: BLE001 - un moteur absent n'arrête rien
            print(f"   {name:22} indisponible : {type(problem).__name__}")
            continue
        if not hypothesis:
            print(f"   {name:22} — rien émis")
            continue
        row = components(truth, hypothesis)
        report = speaker_consistency(truth, hypothesis)
        row["labels"] = len({s for _, _, s in hypothesis})
        row["long_range_300"] = report.long_range_agreement(gap_s=300)
        row["misattributed"] = report.misattributed_share
        row["rtf"] = rtf
        out[name] = row
        print(
            f"   {name:22} {row['der']:6.3f} {row['missed']:7.3f} "
            f"{row['false_alarm']:10.3f} {row['confusion']:10.3f} "
            f"{row['labels']:6} {row['long_range_300']:7.3f} {rtf:6.3f}",
            flush=True,
        )
    return out


@click.command()
@click.option("--ami", "ami_root", default="", help="Le dossier dev-slice d'AMI.")
@click.option("--mix", "mix_stem", default="", help="Un mixage synthétique, sans extension.")
@click.option("--out", "out_path", default="studies/diar_offline_shootout.json")
@click.option("--limit", default=3, help="Nombre de réunions d'AMI.")
def main(ami_root: str, mix_stem: str, out_path: str, limit: int) -> None:
    """Comparer les diariseurs différés sur les deux corpus."""
    relevé: dict[str, dict] = {}

    if mix_stem:
        stem = pathlib.Path(mix_stem)
        truth = reference_from_recipe(json.loads(stem.with_suffix(".json").read_text()))
        clean, sr = sf.read(str(stem.with_suffix(".wav")), dtype="float32")
        if clean.ndim > 1:
            clean = clean.mean(axis=1).astype(np.float32)
        for rt60 in (None, 0.6):
            room = "synthétique sec" if rt60 is None else f"synthétique RT60 {rt60:g} s"
            relevé[room] = evaluate(truth, reverberate(clean, rt60, sr), sr, room)

    if ami_root:
        meetings = sorted(p for p in pathlib.Path(ami_root).iterdir() if (p / "mix.wav").exists())
        for meeting in meetings[:limit]:
            truth = reference_from_rttm(meeting / "words.rttm")
            audio, sr = sf.read(str(meeting / "mix.wav"), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1).astype(np.float32)
            relevé[f"AMI {meeting.name}"] = evaluate(truth, audio, sr, f"AMI {meeting.name}")

    pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    # Le classement se fait sur le PIRE cas, pas sur la moyenne : un défaut se
    # juge sur la condition la plus mal servie, puisque c'est elle qu'un
    # utilisateur rencontrera un jour.
    print("\n── classement, au pire cas de chaque moteur")
    print(f"   {'moteur':22} {'pire DER':>9} {'DER médian':>11} {'pire écart au compte':>21}")
    ranked = []
    for name in contenders():
        rows = [c[name] for c in relevé.values() if name in c]
        if len(rows) < len(relevé):
            continue
        ranked.append(
            (
                max(r["der"] for r in rows),
                float(np.median([r["der"] for r in rows])),
                name,
                max(abs(r["labels"] - 4) for r in rows),
            )
        )
    for worst, median, name, miscount in sorted(ranked):
        print(f"   {name:22} {worst:9.3f} {median:11.3f} {miscount:21}")
    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
