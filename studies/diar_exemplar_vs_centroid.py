"""Un locuteur est-il mieux représenté par un point ou par ses observations ?

Le regroupement incrémental à centroïdes laisse, sur AMI et au seuil qui
minimise le taux d'erreur, vingt étiquettes pour quatre locuteurs. Quatre
paramétrages de seuil ont été essayés et réfutés : ce n'est pas le seuil qui
limite. L'espace d'empreintes n'est pas en cause non plus — la variation d'une
même voix y tourne autour de 0,12 quand la distance entre voix différentes
tourne autour de 0,83, soit un facteur sept.

Reste la représentation. Un centroïde est une moyenne, donc un **point**, et un
point ne peut pas représenter un étalement — or la même voix près et loin du
micro occupe une région, parfois deux modes franchement séparés. Cette étude
compare le centroïde à un ensemble borné d'exemplaires dont la ressemblance est
mesurée au **meilleur point**, et isole les deux effets attendus :

``centroïde``
    La référence, telle qu'elle tourne aujourd'hui.
``exemplaires, sans reprise``
    Même flux, même seuil, mais plus aucune moyenne. Mesure l'apport de la
    représentation seule.
``exemplaires, avec reprise``
    Plus la fusion globale des ensembles, que l'absence de moyenne rend
    définissable — comparer deux ensembles d'observations a un sens, comparer
    deux moyennes mobiles n'en avait pas. C'est le seul levier que le
    diagnostic laissait ouvert contre la sur-segmentation.

L'extraction se fait une fois par réunion et toutes les variantes la rejouent,
donc la grille entière coûte le prix d'un seul passage de modèles.

Usage
-----
    python studies/diar_exemplar_vs_centroid.py --root <dev-slice> --limit 4
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
from vocal_helper.exemplar import ExemplarPolicy
from vocal_helper.incremental import IncrementalPolicy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from diar_incremental_delta_sweep import (  # noqa: E402  (après sys.path, à dessein)
    is_degenerate,
    is_oversegmented,
)
from diar_incremental_vs_greedy import der, reference_from_rttm  # noqa: E402

NEIGHBOURS = (1, 2, 3)
REFERENCES = ("confiance", "médoïde")
#: Créer un locuteur se décide sur UNE observation ; en fusionner deux se décide
#: sur tout ce qu'on a accumulé sur chacun. Une décision mieux informée doit être
#: plus exigeante, pas aussi laxiste — reprendre le seuil de création comme seuil
#: de fusion était une erreur de réglage, et elle coûte 0,795 de taux d'erreur.
MERGE_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.6)


def variants() -> dict[str, object]:
    """Les politiques comparées, nommées comme elles seront lues."""
    out: dict[str, object] = {"centroïde": IncrementalPolicy(delta_new=0.6)}
    for k in NEIGHBOURS:
        out[f"meilleur k={k}"] = ExemplarPolicy(neighbours=k, merge=False)
        out[f"meilleur k={k} + reprise"] = ExemplarPolicy(neighbours=k, merge=True)
    # Un point OBSERVÉ et non un minimum sur un ensemble : pas de moyenne, et
    # pas de biais optimiste non plus, puisque la taille de la mémoire n'entre
    # plus dans la mesure. C'est la règle du magasin biométrique.
    for mode in REFERENCES:
        out[mode] = ExemplarPolicy(reference=mode, merge=False)
    for t in MERGE_THRESHOLDS:
        out[f"médoïde + reprise {t:g}"] = ExemplarPolicy(
            reference="médoïde", merge=True, merge_threshold=t
        )
    return out


@click.command()
@click.option("--root", required=True, help="Le dossier dev-slice d'AMI.")
@click.option("--out", "out_path", default="studies/diar_exemplar_vs_centroid.json")
@click.option("--segmenter", default="sherpa")
@click.option("--limit", default=4, help="Nombre de réunions.")
def main(root: str, out_path: str, segmenter: str, limit: int) -> None:
    """Comparer les représentations à extraction identique."""
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
        print(
            f"\n── {meeting.name} · {duration / 60:.1f} min · {truth} locuteurs"
            f" · {len(windows)} fenêtres"
        )
        print(
            f"   {'variante':26} {'DER':>6} {'dérive':>7} {'mal attr.':>10} "
            f"{'étiq.':>6} {'≥300s':>7}"
        )

        here: dict[str, dict] = {"truth": truth}
        for name, policy in variants().items():
            stage.policy = policy
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
                "oversegmented": is_oversegmented(labels, truth),
            }
            here[name] = row
            mark = " ⚠" if row["degenerate"] else ("  +" if row["oversegmented"] else "")
            print(
                f"   {name:26} {row['der']:6.3f} {row['drift']:7.3f} "
                f"{row['misattributed']:10.3f} {row['labels']:6} "
                f"{row['long_range_300']:7.3f}{mark}",
                flush=True,
            )
        relevé[meeting.name] = here
        pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    print("\n── médianes sur l'ensemble")
    print(f"   {'variante':26} {'DER':>6} {'étiq.':>6} {'≥300s':>7} {'pire DER':>9}")
    for name in variants():
        rows = [m[name] for m in relevé.values() if name in m]
        if not rows:
            continue
        print(
            f"   {name:26} {float(np.median([r['der'] for r in rows])):6.3f} "
            f"{float(np.median([r['labels'] for r in rows])):6.1f} "
            f"{float(np.median([r['long_range_300'] for r in rows])):7.3f} "
            f"{max(r['der'] for r in rows):9.3f}"
        )
    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
