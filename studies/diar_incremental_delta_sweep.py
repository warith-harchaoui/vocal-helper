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

#: Au-delà de ce facteur du vrai compte, une réponse n'est pas « moins bonne »,
#: elle est dégénérée, et son taux d'erreur ne veut plus rien dire. Le garde
#: existe parce que son absence a coûté une conclusion fausse : une réponse
#: effondrée sur une seule étiquette avait été lue comme la meilleure des trois
#: pour l'identité longue portée, alors que son score était le plancher
#: mécanique d'une réponse qui ne distingue personne.
PLAUSIBLE_LABEL_FACTOR = 3.0


def is_degenerate(labels: int, truth: int) -> bool:
    """La réponse est-elle effondrée au point que son score ne veuille rien dire ?

    Le garde est **asymétrique**, et une première version ne l'était pas — à
    tort. Seul l'effondrement flatte le taux d'erreur : une réponse qui met
    tout le monde sur une étiquette est correcte sur la part du locuteur
    dominant et ne distingue personne. La sur-segmentation est l'inverse ; sous
    appariement optimal, les étiquettes en trop ne s'apparient à rien et sont
    comptées comme fautes, donc elle est déjà pénalisée et n'a pas besoin d'être
    écartée. La confondre avec l'effondrement rejetait les bons résultats et
    retenait les mauvais.

    Examples
    --------
    >>> is_degenerate(4, 4)
    False
    >>> is_degenerate(1, 4)
    True
    >>> is_degenerate(11, 4)
    False

    Sur-segmenté n'est pas dégénéré : ce score-là est honnête, et mauvais.

    >>> is_degenerate(122, 4)
    False
    """
    if truth <= 0:
        return False
    return labels * PLAUSIBLE_LABEL_FACTOR < truth


def is_oversegmented(labels: int, truth: int) -> bool:
    """Trop d'étiquettes pour le compte attendu — à signaler, pas à écarter.

    Examples
    --------
    >>> is_oversegmented(122, 4)
    True
    >>> is_oversegmented(11, 4)
    False
    """
    if truth <= 0:
        return False
    return labels > truth * PLAUSIBLE_LABEL_FACTOR


@click.command()
@click.option("--root", required=True, help="Le dossier dev-slice d'AMI.")
@click.option("--out", "out_path", default="studies/diar_incremental_delta_sweep.json")
@click.option("--segmenter", default="sherpa")
@click.option("--limit", default=3, help="Nombre de réunions.")
@click.option("--cap/--no-cap", default=False, help="Plafonner au vrai compte.")
@click.option(
    "--mode",
    type=click.Choice(["absolu", "relatif"]),
    default="absolu",
    help="Seuil absolu, ou fraction de l'échelle inter-locuteurs de l'enregistrement.",
)
def main(root: str, out_path: str, segmenter: str, limit: int, cap: bool, mode: str) -> None:
    """Balayer le seuil de création sur chaque réunion, extraction mise en cache."""
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
        scale = voh.inter_speaker_scale(windows)
        print(
            f"\n── {meeting.name} · {duration / 60:.1f} min · {truth} locuteurs"
            f" · {len(windows)} fenêtres · {embeddings} empreintes"
            f" · échelle inter-locuteurs {scale:.3f}"
            f" · extraction RTF {extraction_rtf:.3f}"
        )
        header = "fraction" if mode == "relatif" else "delta_new"
        print(
            f"   {header:>9} {'seuil':>6} {'DER':>6} {'dérive':>7} {'mal attr.':>10} "
            f"{'étiq.':>6} {'≥300s':>7}"
        )

        here: dict[str, dict] = {}
        for delta in DELTAS:
            cap_at = truth if cap else None
            if mode == "relatif":
                stage.policy = IncrementalPolicy(delta_scale=delta, max_speakers=cap_at)
            else:
                stage.policy = IncrementalPolicy(delta_new=delta, max_speakers=cap_at)
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
            row["degenerate"] = is_degenerate(row["labels"], truth)
            here[f"{delta:.1f}"] = row
            row["threshold"] = stage.policy.delta_new
            flag = " ⚠ dégénéré" if row["degenerate"] else ""
            print(
                f"   {delta:9.1f} {row['threshold']:6.3f} {row['der']:6.3f} "
                f"{row['drift']:7.3f} {row['misattributed']:10.3f} "
                f"{row['labels']:6} {row['long_range_300']:7.3f}{flag}",
                flush=True,
            )
        relevé[meeting.name] = {"truth": truth, "scale": scale, "sweep": here}
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
        degenerate = any(r["degenerate"] for r in rows)
        flag = " ⚠ dégénéré sur au moins une réunion" if degenerate else ""
        print(f"   {delta:9.1f} {median_der:6.3f} {median_labels:6.1f} {median_lr:7.3f}{flag}")
        # Un réglage dégénéré ne concourt pas : son taux d'erreur est flatté par
        # une réponse qui ne distingue personne, et le retenir reproduirait la
        # faute que ce garde-fou existe pour empêcher.
        if not degenerate and (best is None or median_der < best[1]):
            best = (delta, median_der, median_labels)
    if best:
        print(
            f"\n   meilleur delta_new non dégénéré : {best[0]:.1f}"
            f" (DER médian {best[1]:.3f}, {best[2]:.1f} étiquettes)"
        )
    else:
        print("\n   aucun réglage plausible : tous dégénérés")
    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
