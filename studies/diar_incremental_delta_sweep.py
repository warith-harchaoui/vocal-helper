"""Calibrer ``delta_new`` sur du matériel réel.

Le seuil de création d'un locuteur décide à lui seul du nombre de locuteurs
qu'un flux découvre. Sur un mixage synthétique, le défaut de 0,6 trouve
exactement quatre locuteurs pour quatre, parce que les voix y sont propres et
le canal constant. Sur AMI — réunions au micro d'ambiance — le même seuil en
trouve vingt-quatre : la réverbération et la distance au micro écartent les
empreintes d'un *même* locuteur au-delà de 0,6, donc le flux crée au lieu de
rejoindre.

Ce balayage cherche le plateau de fonctionnement sur du réel. Il est abordable
parce que l'extraction (segmentation locale et empreintes, la seule moitié qui
touche aux modèles) se fait une fois par réunion, et que chaque réglage ne
rejoue que le regroupement — du numpy et un algorithme hongrois.

Usage
-----
    python studies/diar_incremental_delta_sweep.py --root <dev-slice> --limit 3
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
from vocal_helper.incremental import IncrementalPolicy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from diar_incremental_vs_greedy import (  # noqa: E402  (après sys.path, à dessein)
    der,
    reference_from_rttm,
)

DELTAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@click.command()
@click.option("--root", required=True, help="Le dossier dev-slice d'AMI.")
@click.option("--out", "out_path", default="studies/diar_incremental_delta_sweep.json")
@click.option("--segmenter", default="sherpa")
@click.option("--limit", default=3, help="Nombre de réunions.")
@click.option("--cap/--no-cap", default=False, help="Plafonner au vrai compte.")
def main(root: str, out_path: str, segmenter: str, limit: int, cap: bool) -> None:
    """Balayer ``delta_new`` sur chaque réunion, extraction mise en cache."""
    meetings = sorted(p for p in pathlib.Path(root).iterdir() if (p / "mix.wav").exists())
    relevé: dict[str, dict] = {}

    for meeting in meetings[:limit]:
        reference = reference_from_rttm(meeting / "words.rttm")
        truth = len({s for _, _, s in reference})
        pcm, sr = sf.read(str(meeting / "mix.wav"), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1).astype(np.float32)
        duration = len(pcm) / sr

        stage = voh.IncrementalDiarStage(segmenter=segmenter)
        started = time.perf_counter()
        windows = stage.extract(pcm, sr)
        extraction_rtf = (time.perf_counter() - started) / duration
        embeddings = sum(len(w.speakers) for w in windows)
        print(
            f"\n── {meeting.name} · {duration / 60:.1f} min · {truth} locuteurs"
            f" · {len(windows)} fenêtres · {embeddings} empreintes"
            f" · extraction RTF {extraction_rtf:.3f}"
        )
        print(
            f"   {'delta_new':>9} {'DER':>6} {'dérive':>7} {'mal attr.':>10} "
            f"{'étiq.':>6} {'≥300s':>7}"
        )

        here: dict[str, dict] = {}
        for delta in DELTAS:
            stage.policy = IncrementalPolicy(delta_new=delta, max_speakers=truth if cap else None)
            stage.reset()
            hypothesis = stage.label(windows, duration)
            report = speaker_consistency(reference, hypothesis)
            row = {
                "der": der(reference, hypothesis),
                "drift": report.worst_drift,
                "misattributed": report.misattributed_share,
                "labels": len({s for _, _, s in hypothesis}),
                "long_range_300": report.long_range_agreement(gap_s=300),
            }
            here[f"{delta:.1f}"] = row
            print(
                f"   {delta:9.1f} {row['der']:6.3f} {row['drift']:7.3f} "
                f"{row['misattributed']:10.3f} {row['labels']:6} "
                f"{row['long_range_300']:7.3f}",
                flush=True,
            )
        relevé[meeting.name] = {"truth": truth, "sweep": here}
        pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    print("\n── médianes sur l'ensemble")
    print(f"   {'delta_new':>9} {'DER':>6} {'étiq.':>6} {'≥300s':>7}")
    best = None
    for delta in DELTAS:
        key = f"{delta:.1f}"
        rows = [m["sweep"][key] for m in relevé.values() if key in m["sweep"]]
        if not rows:
            continue
        median_der = float(np.median([r["der"] for r in rows]))
        median_labels = float(np.median([r["labels"] for r in rows]))
        median_lr = float(np.median([r["long_range_300"] for r in rows]))
        print(f"   {delta:9.1f} {median_der:6.3f} {median_labels:6.1f} {median_lr:7.3f}")
        if best is None or median_der < best[1]:
            best = (delta, median_der, median_labels)
    if best:
        print(
            f"\n   meilleur delta_new : {best[0]:.1f}"
            f" (DER médian {best[1]:.3f}, {best[2]:.1f} étiquettes)"
        )
    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
