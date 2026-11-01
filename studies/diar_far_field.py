"""Qui tient le champ lointain : le modèle de bout en bout ou le diariseur différé ?

Pourquoi AMI ne peut pas répondre
---------------------------------
Sur AMI, Nemotron affiche 0,363 à 0,487 de taux d'erreur quand pyannote donne
0,132 à 0,192 — et pourtant une confusion de locuteurs de 0,003 contre 0,036.
L'écart tient entièrement à de la parole que Nemotron n'émet pas : 616 secondes
sur 815. Deux explications ont été écartées par la mesure, le seuil
d'extraction et le niveau du signal. Restait la matière.

AMI est transcrit sur les **micros-casques** mais on donne au modèle le micro de
salle. L'annotation décrit donc des mots que le signal ne porte pas forcément :
un modèle y est pénalisé pour de la parole inaudible. Et pyannote a AMI dans ses
données d'entraînement. Les deux biais jouent dans le même sens, et aucun des
deux ne se mesure séparément sur ce corpus.

L'instrument juste
------------------
Réverbérer un mixage synthétique supprime les deux. La vérité terrain reste
**exacte par construction** — on n'a pas touché au montage, seulement au canal —
et la seule chose qui varie est la distance au micro. Chaque moteur est alors
comparé à lui-même, du sec au réverbéré, ce qu'aucun corpus ne biaise.

Ce que ça donne
---------------
======================  ========  ========  ============  ========
condition                    DER    manqué     confusion    étiq.
======================  ========  ========  ============  ========
anéchoïque, Nemotron       0,090     0,090         0,000         4
anéchoïque, pyannote       0,299     0,103         0,195         5
RT60 0,3 s, Nemotron       0,105     0,093         0,012         4
RT60 0,3 s, pyannote       0,296     0,110         0,185         4
RT60 0,6 s, Nemotron       0,252     0,101         0,151         4
RT60 0,6 s, pyannote       0,553     0,296         0,249         5
======================  ========  ========  ============  ========

Nemotron gagne dans **toutes** les salles, d'un facteur deux à trois, trouve
partout le bon nombre de locuteurs, et se dégrade plus gracieusement. Et le
point qui tranche la question d'origine : son **manqué reste plat** — 0,090,
0,093, 0,101 — à travers la réverbération. Le champ lointain n'est donc pas ce
qui lui faisait rater un tiers de la parole sur AMI, et il ne reste que
l'annotation pour l'expliquer.

Réserve à garder en tête : tout ceci est synthétique. Le seul enregistrement
réel disponible est AMI, dont l'annotation est justement le problème. Le
verdict tient sur une matière non biaisée mais fabriquée.

Usage
-----
    python studies/diar_far_field.py --mix <chemin/mix_0000>
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from diar_nemotron_threshold import components  # noqa: E402
from rediar_speaker_consistency import reference_from_recipe  # noqa: E402

#: Temps de décroissance de 60 dB. 0,3 s pour une salle de réunion traitée,
#: 0,6 s pour une salle nue aux murs durs.
RT60S: tuple[float | None, ...] = (None, 0.3, 0.6)


def room_impulse(rt60: float, sr: int, seed: int = 0) -> np.ndarray:
    """Une réponse impulsionnelle de salle : bruit à décroissance exponentielle.

    Le modèle le plus simple qui soit juste sur ce qui compte ici — la durée et
    la densité de la queue de réverbération. Un modèle de sources-images
    donnerait des premières réflexions plus réalistes, pour une conclusion que
    la mesure ne distinguerait pas.

    Examples
    --------
    >>> import numpy as np
    >>> rir = room_impulse(0.3, 16000)
    >>> float(round(np.sum(rir**2), 6))
    1.0
    """
    rng = np.random.default_rng(seed)
    length = int(rt60 * 1.5 * sr)
    decay = np.exp(-6.908 * np.arange(length) / (rt60 * sr))
    tail = rng.standard_normal(length) * decay
    tail[0] += 1.0  # le son direct
    return (tail / np.sqrt(np.sum(tail**2))).astype(np.float32)


def reverberate(audio: np.ndarray, rt60: float | None, sr: int) -> np.ndarray:
    """Passer le signal dans la salle, à niveau moyen conservé.

    Conserver le niveau est nécessaire : sans cela on mesurerait l'effet d'un
    gain en même temps que celui de la salle.
    """
    if rt60 is None:
        return audio
    wet = np.convolve(audio, room_impulse(rt60, sr))[: len(audio)]
    before = float(np.sqrt(np.mean(audio**2)))
    after = float(np.sqrt(np.mean(wet**2))) or 1.0
    return (wet * (before / after)).astype(np.float32)


@click.command()
@click.option("--mix", "mix_stem", required=True, help="Le mixage, sans extension.")
@click.option("--out", "out_path", default="studies/diar_far_field.json")
def main(mix_stem: str, out_path: str) -> None:
    """Mesurer chaque moteur du sec au réverbéré."""
    stem = pathlib.Path(mix_stem)
    truth = reference_from_recipe(json.loads(stem.with_suffix(".json").read_text()))
    clean, sr = sf.read(str(stem.with_suffix(".wav")), dtype="float32")
    if clean.ndim > 1:
        clean = clean.mean(axis=1).astype(np.float32)
    print(
        f"{stem.name} · {len(clean) / sr / 60:.1f} min · {len({s for _, _, s in truth})} locuteurs"
    )
    print(
        f"\n   {'salle':12} {'moteur':22} {'DER':>6} {'manqué':>7} "
        f"{'f. alarme':>10} {'confusion':>10} {'étiq.':>6} {'RTF':>6}"
    )
    relevé: dict[str, dict] = {}
    for rt60 in RT60S:
        audio = reverberate(clean, rt60, sr)
        room = "anéchoïque" if rt60 is None else f"RT60 {rt60:g} s"
        for name, build in [
            ("Nemotron ONNX", lambda: voh.NemotronOnnxDiar(mode="low_latency")),
            ("pyannote différé", voh.OfflineDiarStage),
        ]:
            stage = build()
            started = time.perf_counter()
            hypothesis = stage.diarize(audio, sr)
            rtf = (time.perf_counter() - started) / (len(audio) / sr)
            if not hypothesis:
                print(f"   {room:12} {name:22} — rien émis")
                continue
            row = components(truth, hypothesis)
            row["labels"] = len({s for _, _, s in hypothesis})
            row["rtf"] = rtf
            relevé[f"{room} · {name}"] = row
            print(
                f"   {room:12} {name:22} {row['der']:6.3f} {row['missed']:7.3f} "
                f"{row['false_alarm']:10.3f} {row['confusion']:10.3f} "
                f"{row['labels']:6} {rtf:6.3f}",
                flush=True,
            )
        print()
    pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))
    print(f"relevé écrit : {out_path}")


if __name__ == "__main__":
    main()
