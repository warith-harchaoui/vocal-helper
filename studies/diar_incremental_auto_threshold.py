"""Le seuil mesuré sur l'enregistrement tombe-t-il là où il faut ?

Trois tentatives ont échoué avant celle-ci, et c'est ce qui la motive. Un seuil
cosinus absolu ne transfère pas d'une réunion à l'autre. Une fraction de
l'échelle inter-locuteurs **médiane** non plus, parce que deux réunions qui
demandent des seuils différents d'un tiers ont la même médiane — 0,834 et
0,847. Une fraction du **minimum** sépare enfin les deux réunions dans le bon
sens (0,447 contre 0,287) mais sur-corrige largement.

Ce qui manquait à ces trois tentatives est la moitié intra-locuteur du
problème. Ce qui décide d'une fusion n'est ni la distance entre voix
différentes ni la variation d'une voix seule, mais **l'écart entre les deux**.
Les fenêtres voisines se recouvrant de moitié, un locuteur qui parle à la
jointure est embarqué deux fois : la distance entre ces deux vues mesure la
variation d'une voix qui ne change pas, gratuitement et sans vérité terrain.
Le seuil se pose alors entre les deux échelles.

Cette étude vérifie deux choses, et la seconde est la vraie question :
d'abord que les deux échelles sont bien séparées sur du matériel réel, ensuite
que le seuil ainsi posé atteint, sur chaque réunion, le résultat que son
meilleur seuil choisi à la main atteignait.

Usage
-----
    python studies/diar_incremental_auto_threshold.py --root <dev-slice> --limit 4
"""

from __future__ import annotations

import json
import pathlib
import sys

import click
import numpy as np
import soundfile as sf

import vocal_helper as voh
from vocal_helper.consistency import speaker_consistency
from vocal_helper.incremental import IncrementalPolicy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from diar_incremental_delta_sweep import is_degenerate  # noqa: E402  (après sys.path)
from diar_incremental_vs_greedy import der, reference_from_rttm  # noqa: E402

POSITIONS = (0.3, 0.4, 0.5, 0.6, 0.7)
#: Le point de comparaison : le meilleur seuil fixe, cherché finement. C'est le
#: plafond qu'un réglage automatique doit viser, réunion par réunion.
MANUAL = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


@click.command()
@click.option("--root", required=True, help="Le dossier dev-slice d'AMI.")
@click.option("--out", "out_path", default="studies/diar_incremental_auto_threshold.json")
@click.option("--segmenter", default="sherpa")
@click.option("--limit", default=4, help="Nombre de réunions.")
def main(root: str, out_path: str, segmenter: str, limit: int) -> None:
    """Comparer le seuil mesuré au meilleur seuil choisi à la main."""
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
        windows = stage.extract(pcm, sr)
        intra = voh.intra_speaker_scale(windows)
        inter = voh.inter_speaker_scale(windows)
        print(
            f"\n── {meeting.name} · {duration / 60:.1f} min · {truth} locuteurs"
            f" · intra {intra:.3f} · inter {inter:.3f} · écart {inter - intra:+.3f}"
        )

        def evaluate(
            hypothesis: list[tuple[float, float, str]],
            reference: list[tuple[float, float, str]] = reference,
            truth: int = truth,
        ) -> dict:
            report = speaker_consistency(reference, hypothesis)
            labels = len({s for _, _, s in hypothesis})
            return {
                "der": der(reference, hypothesis),
                "drift": report.worst_drift,
                "misattributed": report.misattributed_share,
                "labels": labels,
                "long_range_300": report.long_range_agreement(gap_s=300),
                "degenerate": is_degenerate(labels, truth),
            }

        here: dict[str, dict] = {"intra": intra, "inter": inter, "truth": truth}

        print(f"   {'seuil choisi':>14} {'valeur':>7} {'DER':>6} {'étiq.':>6} {'≥300s':>7}")
        best_manual = None
        for threshold in MANUAL:
            stage.policy = IncrementalPolicy(delta_new=threshold)
            stage.reset()
            row = evaluate(stage.label(windows, duration))
            here[f"manuel/{threshold:g}"] = row
            if not row["degenerate"] and (best_manual is None or row["der"] < best_manual[1]):
                best_manual = (threshold, row["der"], row["labels"])
        if best_manual:
            print(
                f"   {'meilleur manuel':>14} {best_manual[0]:7.3f} "
                f"{best_manual[1]:6.3f} {best_manual[2]:6} "
                f"{here[f'manuel/{best_manual[0]:g}']['long_range_300']:7.3f}"
            )
        else:
            print(f"   {'meilleur manuel':>14} {'—':>7} tous dégénérés")

        for position in POSITIONS:
            stage.policy = IncrementalPolicy(delta_position=position)
            stage.reset()
            row = evaluate(stage.label(windows, duration))
            row["threshold"] = stage.policy.delta_new
            here[f"auto/{position:g}"] = row
            mark = " ⚠" if row["degenerate"] else ""
            print(
                f"   {f'mesuré {position:g}':>14} {row['threshold']:7.3f} "
                f"{row['der']:6.3f} {row['labels']:6} "
                f"{row['long_range_300']:7.3f}{mark}",
                flush=True,
            )
        relevé[meeting.name] = here
        pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    print("\n── le seuil mesuré rattrape-t-il le meilleur seuil manuel ?")
    print(f"   {'position':>9} {'pire DER':>9} {'pire étiq.':>11} {'écart au manuel':>16}")
    for position in POSITIONS:
        key = f"auto/{position:g}"
        rows = [m[key] for m in relevé.values() if key in m]
        if len(rows) < len(relevé):
            continue
        manual_bests = []
        for meeting_rows in relevé.values():
            candidates = [
                row["der"]
                for name, row in meeting_rows.items()
                if isinstance(row, dict) and name.startswith("manuel/") and not row["degenerate"]
            ]
            manual_bests.append(min(candidates) if candidates else float("nan"))
        gaps = [
            row["der"] - best
            for row, best in zip(rows, manual_bests, strict=True)
            if not np.isnan(best)
        ]
        print(
            f"   {position:9g} {max(r['der'] for r in rows):9.3f} "
            f"{max(r['labels'] for r in rows):11} "
            f"{(max(gaps) if gaps else float('nan')):+16.3f}"
        )
    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
