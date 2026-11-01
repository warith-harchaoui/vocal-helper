"""Lequel de nos moteurs gratuits s'accorde le mieux avec le service payant ?

Pourquoi cette mesure
---------------------
Tous les verdicts de cette journée reposent sur des mixages **fabriqués** :
vérité exacte par construction, mais matière artificielle. Le seul
enregistrement réel disponible, AMI, a une annotation transcrite sur les
micros-casques, donc elle décrit de la parole que son micro de salle ne porte
pas — la parole qu'un moteur y rate est 21,8 dB sous celle qu'il émet et 5,4 dB
au-dessus du silence annoté. Ce corpus ne peut donc pas arbitrer.

Il existe une troisième voie : des émissions de télévision déjà diarisées par
**pyannoteAI**, service payant, dont les résultats sont en cache dans le
registre du portefeuille. Cette référence a précisément la propriété qui
manquait — elle n'étiquette que ce que le signal porte, puisqu'elle a été
produite **depuis ce signal**.

Ce que ce chiffre est, et ce qu'il n'est pas
--------------------------------------------
Ce n'est **pas** un taux d'erreur absolu. pyannoteAI n'est pas la vérité : son
propre taux d'erreur est de l'ordre de 0,10 à 0,20, et deux moteurs peuvent
différer d'elle en ayant tous les deux raison. Ce qui est mesuré est un
**désaccord avec un service payant réputé**, sur du matériel réel. C'est la
question que se pose quelqu'un qui hésite à payer, et c'est tout.

Le plafond de huit locuteurs, et pourquoi il faut le voir
---------------------------------------------------------
Sur une heure de plateau, la référence payante trouve de treize à
quarante-deux locuteurs : plateau, reportages, archives, voix off. Nemotron
plafonne à **huit** voies et ne peut donc pas s'accorder, par construction, sur
ce matériel-là. Le nombre de locuteurs de la référence est affiché pour chaque
épisode afin que ce plafond se lise au lieu de se deviner.

Aucun appel payant n'est émis : les références sont lues dans le cache, et
appariées à leur audio par durée.

Usage
-----
    python studies/diar_vs_paid_reference.py --output-root ~/parole-publique/output
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
import time

import click
import numpy as np
import soundfile as sf

import vocal_helper as voh
from vocal_helper.consistency import speaker_consistency

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from diar_nemotron_threshold import components  # noqa: E402

#: Écart de durée en deçà duquel une référence en cache et un audio sont la
#: même séance. Les quelques secondes d'écart viennent du découpage que le
#: service applique à l'envoi.
MATCH_TOLERANCE_S = 30.0


def cached_references(wallet: pathlib.Path) -> list[tuple[float, list[tuple[float, float, str]]]]:
    """Les diarisations payées en cache, avec leur durée. Lecture seule.

    Ouvert en lecture seule à dessein : cette étude ne doit pouvoir **rien**
    dépenser ni rien modifier, et une base ouverte en écriture est une occasion
    de se tromper.
    """
    out = []
    con = sqlite3.connect(f"file:{wallet}?mode=ro", uri=True)
    try:
        for (result,) in con.execute(
            "select result from entries where key like 'pyannoteai.diarize%'"
        ):
            try:
                rows = json.loads(result)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(rows, list) or not rows:
                continue
            spans = [
                (float(r["start"]), float(r["end"]), str(r["speaker"]))
                for r in rows
                if {"start", "end", "speaker"} <= set(r)
            ]
            if spans:
                out.append((max(b for _, b, _ in spans), spans))
    finally:
        con.close()
    return out


def mono_16k(path: pathlib.Path) -> tuple[np.ndarray, int]:
    """Le signal en mono 16 kHz, par audio-helper plutôt qu'à la main."""
    import audio_helper as ah
    import os_helper as osh

    with osh.temporary_folder() as scratch:
        wav = osh.join(scratch, "mono16k.wav")
        ah.sound_converter(str(path), wav, freq=16000, channels=1)
        audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


@click.command()
@click.option("--output-root", required=True, help="Le dossier des épisodes construits.")
@click.option(
    "--wallet",
    default=str(pathlib.Path.home() / ".parole-publique" / "wallet.sqlite"),
    help="Le registre où les diarisations payées sont en cache.",
)
@click.option("--out", "out_path", default="studies/diar_vs_paid_reference.json")
@click.option(
    "--max-ref-speakers",
    default=9,
    help="Ne garder que les séances à ce nombre de locuteurs au plus.",
)
@click.option("--limit", default=2, help="Nombre d'épisodes.")
def main(output_root: str, wallet: str, out_path: str, max_ref_speakers: int, limit: int) -> None:
    """Comparer les moteurs gratuits à la référence payante, sur du réel."""
    references = cached_references(pathlib.Path(wallet).expanduser())
    print(f"{len(references)} diarisations payées en cache")

    episodes = []
    for folder in sorted(pathlib.Path(output_root).expanduser().iterdir()):
        audio_path = folder / "audio.mp3"
        if not audio_path.exists():
            continue
        try:
            duration = sf.info(str(audio_path)).duration
        except Exception:  # noqa: BLE001 - un fichier illisible n'arrête rien
            continue
        best = min(references, key=lambda r: abs(r[0] - duration), default=None)
        if best is None or abs(best[0] - duration) > MATCH_TOLERANCE_S:
            continue
        speakers = len({s for _, _, s in best[1]})
        if speakers > max_ref_speakers:
            continue
        episodes.append((duration, folder, audio_path, best[1], speakers))
    episodes.sort()
    print(
        f"{len(episodes)} épisodes appariés à {max_ref_speakers} locuteurs ou moins "
        f"— le plafond de Nemotron est de huit voies"
    )

    relevé: dict[str, dict] = {}
    engines = {
        "nemotron": lambda: voh.NemotronOnnxDiar(),
        "pyannote community-1": lambda: voh.OfflineDiarStage(backend="pyannote"),
        "nemo sortformer": lambda: voh.OfflineDiarStage(backend="nemo"),
        "sherpa": lambda: voh.OfflineDiarStage(backend="sherpa"),
    }

    for duration, folder, audio_path, reference, ref_speakers in episodes[:limit]:
        audio, sr = mono_16k(audio_path)
        print(
            f"\n── {folder.name[:58]}"
            f"\n   {duration / 60:.1f} min · la référence payante y voit "
            f"{ref_speakers} locuteurs"
        )
        print(
            f"   {'moteur':22} {'désaccord':>10} {'manqué':>7} {'f. alarme':>10} "
            f"{'confusion':>10} {'étiq.':>6} {'≥300s':>7} {'RTF':>6}"
        )
        here: dict[str, dict] = {"ref_speakers": ref_speakers, "duration_s": duration}
        for name, build in engines.items():
            try:
                stage = build()
                started = time.perf_counter()
                hypothesis = stage.diarize(audio, sr)
                rtf = (time.perf_counter() - started) / (len(audio) / sr)
            except Exception as problem:  # noqa: BLE001
                print(f"   {name:22} indisponible : {type(problem).__name__}")
                continue
            if not hypothesis:
                print(f"   {name:22} — rien émis")
                continue
            row = components(reference, hypothesis)
            report = speaker_consistency(reference, hypothesis)
            row["labels"] = len({s for _, _, s in hypothesis})
            row["long_range_300"] = report.long_range_agreement(gap_s=300)
            row["rtf"] = rtf
            here[name] = row
            print(
                f"   {name:22} {row['der']:10.3f} {row['missed']:7.3f} "
                f"{row['false_alarm']:10.3f} {row['confusion']:10.3f} "
                f"{row['labels']:6} {row['long_range_300']:7.3f} {rtf:6.3f}",
                flush=True,
            )
        relevé[folder.name] = here
        pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    print(
        "\n   Rappel de lecture : ce chiffre est un DÉSACCORD avec un service payant,"
        "\n   pas un taux d'erreur absolu. pyannoteAI n'est pas la vérité."
    )
    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
