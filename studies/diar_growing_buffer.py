"""Diariser le tampon entier, périodiquement : la qualité du différé, en direct.

L'idée
------
Deux mesures de la même journée se combinent mieux qu'elles ne l'avaient laissé
voir. D'une part, l'identité voyage très bien par **recouvrement temporel**
entre deux diarisations — c'est le dispositif éprouvé dans
``diar_overlap_identity_transport.py``, dont la perte tient entièrement à son
amorçage synthétique. D'autre part, le diariseur différé coûte **RTF 0,039**.

Or l'amorçage n'existe que pour un cas : un locuteur absent de la zone commune
aux deux diarisations. Si la seconde porte sur **tout l'audio depuis le début**
et non sur un morceau, ce cas disparaît — quiconque a parlé est dans la zone
commune, qui couvre alors la quasi-totalité du signal. L'appariement devient
presque exact, et aucun son synthétique n'est nécessaire.

Le coût se calcule. Rafraîchir aux instants t, 2t, 4t… diarise au total environ
``2 × T`` d'audio pour une séance de durée ``T``, donc un RTF d'environ
``2 × 0,039 ≈ 0,08`` — et la **dernière passe est une diarisation du fichier
entier**, donc la qualité finale est celle du différé, atteinte en direct.

Ce qui se paie, ce n'est pas la qualité, c'est la **fraîcheur** : la parole
postérieure à la dernière passe n'a pas encore d'identité faisant autorité. Un
étage bon marché la nomme provisoirement, et chaque passe publie ses
corrections. Cette étude mesure donc les trois grandeurs qui décident :
la qualité finale, le retard avant qu'un passage reçoive son identité
définitive, et le coût.

Les étiquettes d'une passe sont arbitraires — c'est l'appariement optimal
contre la vérité terrain qui juge — mais elles doivent rester **stables pour le
lecteur** d'une passe à l'autre. D'où l'appariement hongrois entre passes
successives sur leur préfixe commun, et le décompte des révisions qu'il
entraîne.

Usage
-----
    python studies/diar_growing_buffer.py --root <dev-slice> --limit 2
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
from diar_incremental_vs_greedy import der, reference_from_rttm  # noqa: E402
from diar_overlap_identity_transport import best_bijection, truth_on  # noqa: E402


def geometric_schedule(duration_s: float, *, first_s: float, ratio: float) -> list[float]:
    """Les instants de rafraîchissement, espacés géométriquement.

    Un pas constant ferait croître le coût comme le carré de la durée ; un pas
    géométrique le garde proportionnel, parce que la somme des préfixes
    diarisés reste un petit multiple de la durée totale.

    Examples
    --------
    >>> geometric_schedule(100.0, first_s=10.0, ratio=2.0)
    [10.0, 20.0, 40.0, 80.0, 100.0]
    >>> geometric_schedule(8.0, first_s=10.0, ratio=2.0)
    [8.0]
    """
    out: list[float] = []
    t = first_s
    while t < duration_s:
        out.append(round(t, 6))
        t *= ratio
    out.append(round(duration_s, 6))
    return out


@click.command()
@click.option("--root", required=True, help="Le dossier dev-slice d'AMI.")
@click.option("--out", "out_path", default="studies/diar_growing_buffer.json")
@click.option("--segmenter", default="pyannote")
@click.option("--first-s", default=60.0, help="Première passe à cet instant.")
@click.option("--ratio", default=1.6, help="Facteur d'espacement des passes.")
@click.option("--limit", default=2, help="Nombre de réunions.")
def main(
    root: str, out_path: str, segmenter: str, first_s: float, ratio: float, limit: int
) -> None:
    """Mesurer qualité finale, fraîcheur et coût du rafraîchissement sur tampon entier."""
    meetings = sorted(p for p in pathlib.Path(root).iterdir() if (p / "mix.wav").exists())
    engine = voh.OfflineDiarStage(backend=segmenter)
    relevé: dict[str, dict] = {}

    for meeting in meetings[:limit]:
        truth = reference_from_rttm(meeting / "words.rttm")
        n_truth = len({s for _, _, s in truth})
        pcm, sr = sf.read(str(meeting / "mix.wav"), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1).astype(np.float32)
        duration = len(pcm) / sr
        schedule = geometric_schedule(duration, first_s=first_s, ratio=ratio)
        print(
            f"\n── {meeting.name} · {duration / 60:.1f} min · {n_truth} locuteurs"
            f" · {len(schedule)} passes à {', '.join(f'{t:.0f}' for t in schedule)} s"
        )
        print(
            f"   {'passe':>6} {'jusqu’à':>8} {'DER préfixe':>12} {'étiq.':>6} "
            f"{'renommés':>9} {'révisions':>10} {'RTF cumulé':>11}"
        )

        previous: list[tuple[float, float, str]] = []
        stable: dict[str, str] = {}
        minted = 0
        diarized_s = 0.0
        spent = 0.0
        passes: list[dict] = []
        final: list[tuple[float, float, str]] = []

        for index, upto in enumerate(schedule, start=1):
            buffer = pcm[: int(upto * sr)]
            started = time.perf_counter()
            raw = engine.diarize(buffer, sr)
            spent += time.perf_counter() - started
            diarized_s += upto
            if not raw:
                continue

            # Rendre les étiquettes de cette passe stables pour le lecteur, en
            # les appariant à la passe précédente sur leur préfixe commun — qui
            # est ici presque tout l'audio, d'où un appariement quasi exact.
            bridge = best_bijection(raw, previous) if previous else {}
            mapping: dict[str, str] = {}
            for label in sorted({lab for _, _, lab in raw}):
                carried = bridge.get(label)
                if carried is not None and carried in stable:
                    mapping[label] = stable[carried]
                else:
                    mapping[label] = f"P{minted}"
                    minted += 1
            renamed = [(t0, t1, mapping[lab]) for t0, t1, lab in raw]

            revisions = 0
            if final:
                for t0, t1, name in final:
                    overlap = [
                        (min(t1, b1) - max(t0, b0), new)
                        for b0, b1, new in renamed
                        if min(t1, b1) - max(t0, b0) > 0
                    ]
                    if overlap and max(overlap)[1] != name:
                        revisions += 1

            prefix_truth = truth_on(truth, 0.0, upto)
            row = {
                "upto_s": upto,
                "der": der(prefix_truth, renamed) if prefix_truth else float("nan"),
                "labels": len({lab for _, _, lab in renamed}),
                "renamed": sum(1 for lab, name in mapping.items() if name.startswith("P")),
                "revisions": revisions,
                "rtf_cumulative": spent / upto,
            }
            passes.append(row)
            print(
                f"   {index:6} {upto:8.0f} {row['der']:12.3f} {row['labels']:6} "
                f"{row['renamed']:9} {revisions:10} {row['rtf_cumulative']:11.3f}",
                flush=True,
            )

            previous = renamed
            stable = {name: name for name in mapping.values()}
            final = renamed

        report = speaker_consistency(truth, sorted(final))
        here = {
            "truth": n_truth,
            "duration_s": duration,
            "passes": passes,
            "final_der": der(truth, sorted(final)),
            "final_labels": len({s for _, _, s in final}),
            "final_long_range_300": report.long_range_agreement(gap_s=300),
            "final_misattributed": report.misattributed_share,
            "audio_diarized_s": diarized_s,
            "rtf_total": spent / duration,
            "worst_staleness_s": max(
                (b - a for a, b in zip([0.0, *schedule], schedule, strict=False)), default=0.0
            ),
        }
        print(
            f"   final : DER {here['final_der']:.3f} · {here['final_labels']} étiquettes"
            f" · accord ≥300 s {here['final_long_range_300']:.3f}"
        )
        print(
            f"   coût : {diarized_s / duration:.2f}× l'audio diarisé"
            f" · RTF total {here['rtf_total']:.3f}"
            f" · pire fraîcheur {here['worst_staleness_s']:.0f} s"
        )
        relevé[meeting.name] = here
        pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
