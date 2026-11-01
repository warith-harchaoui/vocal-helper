"""Existe-t-il un seul réglage de seuil valable pour tous les enregistrements ?

C'est la question qui bloque le direct gratuit. Un seuil cosinus **absolu** ne
transfère pas : 0,8 donne le meilleur taux d'erreur sur ES2011a et fusionne
deux vrais locuteurs sur ES2011b. Exprimer le seuil comme une fraction de
l'échelle inter-locuteurs **médiane** n'y change rien, parce que les deux
réunions ont justement la même médiane — 0,834 et 0,847, indiscernables.

Ce qui les sépare est la **paire la plus confusable**, que la médiane noie.
D'où ce balayage croisé : pour chaque réunion, une seule extraction, puis une
grille quantile × fraction rejouée dessus. On cherche un couple unique dont le
pire résultat sur l'ensemble des réunions reste acceptable — un réglage qui
marche en moyenne mais casse une réunion sur trois ne vaut rien en production.

Usage
-----
    python studies/diar_incremental_scale_calibration.py --root <dev-slice> --limit 3
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

QUANTILES = (0.0, 0.05, 0.1, 0.25, 0.5)
FRACTIONS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4)


@click.command()
@click.option("--root", required=True, help="Le dossier dev-slice d'AMI.")
@click.option("--out", "out_path", default="studies/diar_incremental_scale_calibration.json")
@click.option("--segmenter", default="sherpa")
@click.option("--limit", default=3, help="Nombre de réunions.")
def main(root: str, out_path: str, segmenter: str, limit: int) -> None:
    """Balayer la grille quantile × fraction, une extraction par réunion."""
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
        scales = {q: voh.inter_speaker_scale(windows, quantile=q) for q in QUANTILES}
        print(
            f"\n── {meeting.name} · {duration / 60:.1f} min · {truth} locuteurs"
            f" · {len(windows)} fenêtres"
        )
        print("   échelles : " + " · ".join(f"q{q:g}={scales[q]:.3f}" for q in QUANTILES))

        here: dict[str, dict] = {}
        for quantile in QUANTILES:
            cells = []
            for fraction in FRACTIONS:
                stage.policy = IncrementalPolicy(delta_scale=fraction, delta_quantile=quantile)
                stage.reset()
                hypothesis = stage.label(windows, duration)
                report = speaker_consistency(reference, hypothesis)
                labels = len({s for _, _, s in hypothesis})
                row = {
                    "der": der(reference, hypothesis),
                    "drift": report.worst_drift,
                    "misattributed": report.misattributed_share,
                    "labels": labels,
                    "long_range_300": report.long_range_agreement(gap_s=300),
                    "degenerate": is_degenerate(labels, truth),
                    "threshold": stage.policy.delta_new,
                }
                here[f"q{quantile:g}/f{fraction:g}"] = row
                mark = "!" if row["degenerate"] else " "
                cells.append(f"{row['der']:.3f}{mark}{row['labels']:<3}")
            print(f"   q{quantile:<5g} " + " ".join(cells), flush=True)
        print("   fractions " + " ".join(f"{f:<8g}" for f in FRACTIONS))
        relevé[meeting.name] = {"truth": truth, "scales": scales, "grid": here}
        pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    # Le critère : le PIRE résultat du couple sur l'ensemble des réunions. Un
    # réglage se juge sur sa réunion la plus mal servie, pas sur sa moyenne.
    print("\n── un seul couple pour toutes les réunions ?")
    print(f"   {'couple':14} {'pire DER':>9} {'pire étiq.':>11} {'dégénéré':>9}")
    ranked = []
    for quantile in QUANTILES:
        for fraction in FRACTIONS:
            key = f"q{quantile:g}/f{fraction:g}"
            rows = [m["grid"][key] for m in relevé.values() if key in m["grid"]]
            if len(rows) < len(relevé):
                continue
            worst_der = max(r["der"] for r in rows)
            worst_labels = max(r["labels"] for r in rows)
            any_degenerate = any(r["degenerate"] for r in rows)
            ranked.append((worst_der, key, worst_labels, any_degenerate))
    for worst_der, key, worst_labels, degenerate in sorted(ranked)[:8]:
        print(f"   {key:14} {worst_der:9.3f} {worst_labels:11} {'oui' if degenerate else 'non':>9}")
    clean = [r for r in sorted(ranked) if not r[3]]
    if clean:
        worst_der, key, worst_labels, _ = clean[0]
        print(
            f"\n   meilleur couple non dégénéré : {key}"
            f" (pire DER {worst_der:.3f}, pire compte {worst_labels})"
        )
    else:
        print("\n   aucun couple ne reste plausible sur toutes les réunions")
    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
