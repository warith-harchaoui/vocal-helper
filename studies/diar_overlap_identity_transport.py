"""L'identité peut-elle voyager par le temps seul, sans aucune empreinte ?

La question
-----------
Le regroupement par empreintes bute sur un plafond : sur AMI, l'étage en flux
plafonne à 0,23 de taux d'erreur avec vingt étiquettes pour quatre locuteurs,
et aucun réglage de seuil n'y change rien — quatre paramétrages essayés, quatre
réfutés. D'où l'idée à sonder ici, qui se passe entièrement de géométrie
d'empreintes.

Le morceau courant est diarisé **deux fois**. Une fois seul, ce qui donne des
étiquettes locales et des frontières. Une fois précédé d'un **amorçage
synthétique** où chaque locuteur déjà connu occupe un créneau de position
*connue*. Dans cette seconde passe, relire quelle étiquette couvre quel créneau
nomme les grappes ; puis le **recouvrement temporel** entre les deux
diarisations, sur la zone commune — le morceau lui-même — transporte ces noms
vers la première. L'identité voyage par le temps.

Le point fort du dispositif est que l'amorçage garantit la présence de **tous**
les locuteurs connus dans la seconde passe, même ceux qui n'ont pas parlé dans
le morceau. Un simple recouvrement entre morceaux consécutifs ne sait pas faire
ça : il ne transporte que les locuteurs présents dans la zone partagée.

Les deux appariements en jeu sont des **bijections** entre grappes, donc tous
deux résolus par algorithme hongrois. Les noms de grappes sont libres — c'est
l'appariement optimal contre la vérité terrain qui juge — mais leur **nombre**
ne l'est pas : une étiquette en trop ne s'apparie à rien et compte comme faute.

Ce que l'étude sépare
---------------------
``hors ligne``
    Le fichier entier d'un coup : la borne supérieure.
``transport``
    Le dispositif décrit ci-dessus.
``direct``
    Les étiquettes de la **seconde** passe sur la zone du morceau, nommées par
    la relecture, sans passer par la première. Si c'est aussi bon, la première
    diarisation est superflue et le dispositif se simplifie.
``oracle``
    Les étiquettes de la première passe appariées **optimalement à la vérité
    terrain**, morceau par morceau. Aucun transport réel ne peut faire mieux,
    donc l'écart entre ``oracle`` et ``hors ligne`` mesure ce que coûte le
    découpage en morceaux, et l'écart entre ``transport`` et ``oracle`` mesure
    ce que coûte le transport. C'est cette séparation qui tranche : elle dit si
    l'idée échoue, ou si c'est la diarisation du morceau qui plafonne.

Usage
-----
    python studies/diar_overlap_identity_transport.py --root <dev-slice> --limit 2
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import click
import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

import vocal_helper as voh
from vocal_helper.consistency import speaker_consistency
from vocal_helper.rediar import (
    assemble_primer,
    build_references,
    harvest_room_tone,
    read_back_identities,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from diar_incremental_vs_greedy import der, reference_from_rttm  # noqa: E402


def overlap_matrix(
    left: list[tuple[float, float, str]], right: list[tuple[float, float, str]]
) -> tuple[list[str], list[str], np.ndarray]:
    """Temps partagé entre chaque étiquette de gauche et chaque étiquette de droite.

    Examples
    --------
    >>> rows, cols, m = overlap_matrix([(0.0, 2.0, "a")], [(1.0, 4.0, "x")])
    >>> rows, cols, float(m[0, 0])
    (['a'], ['x'], 1.0)
    """
    rows = sorted({label for _, _, label in left})
    cols = sorted({label for _, _, label in right})
    matrix = np.zeros((len(rows), len(cols)), dtype=np.float64)
    ri = {label: i for i, label in enumerate(rows)}
    ci = {label: j for j, label in enumerate(cols)}
    for a0, a1, a in left:
        for b0, b1, b in right:
            shared = min(a1, b1) - max(a0, b0)
            if shared > 0:
                matrix[ri[a], ci[b]] += shared
    return rows, cols, matrix


def best_bijection(
    left: list[tuple[float, float, str]], right: list[tuple[float, float, str]]
) -> dict[str, str]:
    """La correspondance un-pour-un entre deux diarisations qui maximise le temps partagé.

    Un argmax par étiquette conclurait localement bien et globalement mal, et
    pourrait surtout envoyer deux étiquettes de gauche sur la même à droite, ce
    qui fusionnerait deux locuteurs sans le dire.

    Examples
    --------
    >>> best_bijection([(0.0, 2.0, "a"), (2.0, 4.0, "b")],
    ...                [(0.0, 2.0, "y"), (2.0, 4.0, "x")])
    {'a': 'y', 'b': 'x'}
    """
    rows, cols, matrix = overlap_matrix(left, right)
    if not rows or not cols:
        return {}
    out: dict[str, str] = {}
    ri, ci = linear_sum_assignment(-matrix)
    for i, j in zip(ri.tolist(), ci.tolist(), strict=True):
        if matrix[i, j] > 0:
            out[rows[i]] = cols[j]
    return out


def shift(
    segments: list[tuple[float, float, str]], *, by: float, keep_from: float
) -> list[tuple[float, float, str]]:
    """Ramener les segments postérieurs à ``keep_from`` sur une base décalée."""
    out = []
    for t0, t1, label in segments:
        if t1 <= keep_from:
            continue
        out.append((max(t0, keep_from) - by, t1 - by, label))
    return out


def truth_on(
    reference: list[tuple[float, float, str]], t0: float, t1: float
) -> list[tuple[float, float, str]]:
    """La vérité terrain restreinte à une fenêtre, en temps relatifs."""
    out = []
    for a0, a1, speaker in reference:
        lo, hi = max(a0, t0), min(a1, t1)
        if hi > lo:
            out.append((lo - t0, hi - t0, speaker))
    return out


@click.command()
@click.option("--root", required=True, help="Le dossier dev-slice d'AMI.")
@click.option("--out", "out_path", default="studies/diar_overlap_identity_transport.json")
@click.option("--segmenter", default="pyannote", help="Moteur de diarisation des passes.")
@click.option("--chunk-s", default=60.0, help="Longueur d'un morceau.")
@click.option("--budget-s", default=8.0, help="Durée de référence par locuteur dans l'amorçage.")
@click.option("--limit", default=2, help="Nombre de réunions.")
@click.option(
    "--refresh/--freeze",
    default=False,
    help="Remplacer une référence par un échantillon plus long, ou la figer à la première vue.",
)
def main(
    root: str,
    out_path: str,
    segmenter: str,
    chunk_s: float,
    budget_s: float,
    limit: int,
    refresh: bool,
) -> None:
    """Mesurer le transport d'identité par recouvrement temporel."""
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
        print(
            f"\n── {meeting.name} · {duration / 60:.1f} min · {n_truth} locuteurs"
            f" · morceaux de {chunk_s:g} s · amorçage {budget_s:g} s/locuteur"
            f" · références {'rafraîchies' if refresh else 'figées'}"
        )

        room = harvest_room_tone(pcm[: int(min(duration, 120.0) * sr)], sr, 0.8)
        references: dict[str, np.ndarray] = {}
        produced: dict[str, list[tuple[float, float, str]]] = {
            "transport": [],
            "direct": [],
            "oracle": [],
        }
        placed_total = known_total = 0
        carried_total = labels_total = 0
        started = time.perf_counter()
        minted = 0

        n_chunks = max(1, int(np.ceil(duration / chunk_s)))
        for k in range(n_chunks):
            t0 = k * chunk_s
            t1 = min(duration, t0 + chunk_s)
            if t1 - t0 < 5.0:
                continue
            chunk = pcm[int(t0 * sr) : int(t1 * sr)]
            first = engine.diarize(chunk, sr)
            if not first:
                continue
            labels_total += len({label for _, _, label in first})

            # --- oracle : la meilleure correspondance possible vers la vérité
            here_truth = truth_on(truth, t0, t1)
            oracle_map = best_bijection(first, here_truth)
            for a0, a1, label in first:
                name = oracle_map.get(label)
                if name is not None:
                    produced["oracle"].append((t0 + a0, t0 + a1, name))

            # --- transport et direct
            mapping: dict[str, str] = {}
            direct: list[tuple[float, float, str]] = []
            if references:
                primer, slots = assemble_primer(references, room, sr)
                lead = len(primer) / sr
                second = engine.diarize(np.concatenate([primer, chunk]).astype(np.float32), sr)
                back = read_back_identities(second, slots)
                known_total += len(slots)
                placed_total += len(set(back.values()))
                on_chunk = shift(second, by=lead, keep_from=lead)
                for b0, b1, label in on_chunk:
                    name = back.get(label)
                    if name is not None:
                        direct.append((t0 + b0, t0 + b1, name))
                bridge = best_bijection(first, on_chunk)
                mapping = {one: back[two] for one, two in bridge.items() if two in back}
                carried_total += len(mapping)
            produced["direct"].extend(direct)

            for a0, a1, label in first:
                name = mapping.get(label)
                if name is None:
                    name = f"N{minted}"
                    mapping[label] = name
                    minted += 1
                produced["transport"].append((t0 + a0, t0 + a1, name))

            # --- récolter une référence pour chaque identité vue dans ce morceau
            renamed = [(a0, a1, mapping[label]) for a0, a1, label in first if label in mapping]
            fresh = build_references(
                pcm=chunk, sample_rate=sr, segments=renamed, offset_s=0.0, budget_s=budget_s
            )
            # Figer une référence sur les premières secondes jamais entendues est
            # ce qui plafonnait la relecture : sur une réunion dont le moteur ne
            # trouve que trois locuteurs sur quatre, le premier morceau n'a aucune
            # identité à laquelle se raccrocher, donc la référence récoltée peut
            # être un mélange de deux voix — et elle le reste pour toute la séance.
            for name, sample in fresh.items():
                if not len(sample):
                    continue
                held = references.get(name)
                if held is None or (refresh and len(sample) > len(held)):
                    references[name] = sample

        elapsed = time.perf_counter() - started
        print(
            f"   {'condition':12} {'DER':>6} {'dérive':>7} {'mal attr.':>10} "
            f"{'étiq.':>6} {'≥300s':>7}"
        )
        here: dict[str, dict] = {"truth": n_truth, "rtf": elapsed / duration}
        conditions = dict(produced)
        conditions["hors ligne"] = engine.diarize(pcm, sr)
        for name in ("hors ligne", "oracle", "transport", "direct"):
            rows = sorted(conditions[name])
            if not rows:
                print(f"   {name:12} —")
                continue
            report = speaker_consistency(truth, rows)
            row = {
                "der": der(truth, rows),
                "drift": report.worst_drift,
                "misattributed": report.misattributed_share,
                "labels": len({s for _, _, s in rows}),
                "long_range_300": report.long_range_agreement(gap_s=300),
            }
            here[name] = row
            print(
                f"   {name:12} {row['der']:6.3f} {row['drift']:7.3f} "
                f"{row['misattributed']:10.3f} {row['labels']:6} "
                f"{row['long_range_300']:7.3f}",
                flush=True,
            )
        here["placement"] = placed_total / known_total if known_total else float("nan")
        here["carried"] = carried_total / labels_total if labels_total else float("nan")
        here["minted"] = minted
        here["refresh"] = refresh
        here["chunk_s"] = chunk_s
        here["budget_s"] = budget_s
        print(
            f"   relecture : {here['placement']:.1%} des créneaux placés"
            f" · transport : {here['carried']:.1%} des étiquettes nommées"
            f" · {minted} identités créées · RTF {here['rtf']:.3f}"
        )
        relevé[meeting.name] = here
        pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
