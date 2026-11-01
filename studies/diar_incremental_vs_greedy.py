"""diart porté contre le plus-proche-centroïde glouton, sur matériel réel.

Le regroupement incrémental sous contrainte (:mod:`vocal_helper.incremental`)
bat le diariseur différé pleine fenêtre sur un mixage synthétique, en devinant
le bon nombre de locuteurs. Un mixage fabriqué est un instrument commode mais
complaisant : les voix y sont propres, le canal constant par locuteur, le
recouvrement maîtrisé. Cette campagne refait la mesure sur AMI, des réunions
réelles enregistrées au micro d'ambiance — réverbération, parasites, voix qui
se marchent dessus.

Trois conditions, toutes sans compte fourni, puisque deviner le nombre de
locuteurs fait partie de ce qu'on mesure :

``différé``
    :class:`~vocal_helper.OfflineDiarStage`, le défaut de production en lot.
``en ligne glouton``
    :class:`~vocal_helper.OnlineDiarStage`, le défaut actuel en direct.
``diart porté``
    :class:`~vocal_helper.IncrementalDiarStage`, segmenteur au choix.

Usage
-----
    python studies/diar_incremental_vs_greedy.py --root <dev-slice> --out relevé.json
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import time

import click
import numpy as np
import soundfile as sf

import vocal_helper as voh
from vocal_helper.consistency import speaker_consistency


def reference_from_rttm(path: pathlib.Path) -> list[tuple[float, float, str]]:
    """Lire un RTTM en ``[(t0, t1, locuteur), …]`` trié par début."""
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != "SPEAKER":
            continue
        start, duration = float(parts[3]), float(parts[4])
        if duration > 0:
            rows.append((start, start + duration, parts[7]))
    return sorted(rows)


def der(reference, hypothesis) -> float:
    """DER avec appariement optimal, collier 0,25 s — la convention du dépôt."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def to_annotation(rows):
        annotation = Annotation()
        for t0, t1, speaker in rows:
            if t1 > t0:
                annotation[Segment(t0, t1)] = speaker
        return annotation

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    return float(metric(to_annotation(reference), to_annotation(hypothesis)))


async def _greedy(pcm: np.ndarray, sr: int) -> list[tuple[float, float, str]]:
    """L'étage glouton, alimenté comme la vraie chaîne l'alimente.

    L'étage ne consomme pas des trames brutes mais des **segments voisés** : il
    est en aval d'un VAD. Le lui donner autrement fausserait la comparaison en
    sa défaveur, donc on fait passer l'audio par le même VAD Silero que le
    direct, puis on lui sert ses segments.
    """
    inbox: asyncio.Queue = asyncio.Queue()
    voiced: asyncio.Queue = asyncio.Queue()
    frame = int(0.032 * sr)
    for i in range(0, pcm.size, frame):
        inbox.put_nowait({"t0": i / sr, "sample_rate": sr, "pcm": pcm[i : i + frame]})
    inbox.put_nowait(None)
    await voh.SileroVADStage().run(inbox, voiced)

    sink: asyncio.Queue = asyncio.Queue()
    await voh.OnlineDiarStage().run(voiced, sink)
    out = []
    while not sink.empty():
        item = sink.get_nowait()
        if item is None:
            continue
        speaker = item["speaker"] if isinstance(item, dict) else item.speaker
        t0 = item["t0"] if isinstance(item, dict) else item.t0
        t1 = item["t1"] if isinstance(item, dict) else item.t1
        if speaker != "S?":
            out.append((t0, t1, speaker))
    return sorted(out)


@click.command()
@click.option("--root", required=True, help="Le dossier dev-slice d'AMI.")
@click.option("--out", "out_path", default="studies/diar_incremental_vs_greedy.json")
@click.option("--segmenter", default="sherpa", help="Segmenteur local de diart porté.")
@click.option("--limit", default=0, help="Ne traiter que les N premières réunions.")
def main(root: str, out_path: str, segmenter: str, limit: int) -> None:
    """Mesurer les trois conditions sur chaque réunion du dossier."""
    meetings = sorted(p for p in pathlib.Path(root).iterdir() if (p / "mix.wav").exists())
    if limit:
        meetings = meetings[:limit]
    relevé: dict[str, dict] = {}

    for meeting in meetings:
        reference = reference_from_rttm(meeting / "words.rttm")
        truth = len({s for _, _, s in reference})
        pcm, sr = sf.read(str(meeting / "mix.wav"), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1).astype(np.float32)
        duration = len(pcm) / sr
        print(f"\n── {meeting.name} · {duration / 60:.1f} min · {truth} locuteurs")
        print(
            f"   {'condition':20} {'DER':>6} {'dérive':>7} {'mal attr.':>10} "
            f"{'étiq.':>6} {'≥300s':>7} {'RTF':>6}"
        )
        here: dict[str, dict] = {"duration_s": duration, "truth": truth}

        # Les trois conditions lient pcm/sr explicitement : une lambda qui
        # capture la variable de boucle est un piège, même appelée aussitôt.
        conditions = {
            "différé": lambda p=pcm, r=sr: voh.OfflineDiarStage().diarize(p, r),
            "en ligne glouton": lambda p=pcm, r=sr: asyncio.run(_greedy(p, r)),
            "diart porté": lambda p=pcm, r=sr: voh.IncrementalDiarStage(
                segmenter=segmenter
            ).diarize(p, r),
        }
        for name, run in conditions.items():
            started = time.perf_counter()
            try:
                hypothesis = run()
            except Exception as problem:  # une condition qui tombe ne doit pas tout perdre
                print(f"   {name:20} échec : {type(problem).__name__}: {problem}")
                here[name] = {"error": f"{type(problem).__name__}: {problem}"}
                continue
            rtf = (time.perf_counter() - started) / duration
            report = speaker_consistency(reference, hypothesis)
            row = {
                "der": der(reference, hypothesis),
                "drift": report.worst_drift,
                "misattributed": report.misattributed_share,
                "labels": len({s for _, _, s in hypothesis}),
                "long_range_300": report.long_range_agreement(gap_s=300),
                "rtf": rtf,
            }
            here[name] = row
            print(
                f"   {name:20} {row['der']:6.3f} {row['drift']:7.3f} "
                f"{row['misattributed']:10.3f} {row['labels']:6} "
                f"{row['long_range_300']:7.3f} {rtf:6.3f}",
                flush=True,
            )
        relevé[meeting.name] = here
        pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    print("\n── médianes sur l'ensemble")
    print(f"   {'condition':20} {'DER':>6} {'étiq.':>6} {'≥300s':>7} {'RTF':>6}")
    for name in ("différé", "en ligne glouton", "diart porté"):
        rows = [m[name] for m in relevé.values() if name in m and "der" in m[name]]
        if not rows:
            continue

        def med(key: str, rows: list[dict] = rows) -> float:
            return float(np.median([r[key] for r in rows]))

        print(
            f"   {name:20} {med('der'):6.3f} {med('labels'):6.1f} "
            f"{med('long_range_300'):7.3f} {med('rtf'):6.3f}"
        )
    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
