"""La cohérence des locuteurs sur toute la durée, mesurée sur son synthétique.

Question
--------
La re-diarisation en direct existe pour une raison précise : tenir l'identité d'une voix
d'un bout à l'autre, à travers des fenêtres qui ne se voient pas. Le DER ne mesure pas ça.
Sur un long enregistrement il est dominé par les **frontières** : des milliers de petites
erreurs de bord noient la seule panne qui ruine vraiment un relevé, qui est un locuteur
perdant son identité à mi-parcours et finissant la réunion sous une autre étiquette.

Instrument
----------
Le fabricateur **bagarre** (`pdbms/scripts/fabricate_bagarre.py`) : chaque tour est découpé
dans un clip mono-locuteur connu, donc l'identité est **exacte par construction** et non par
annotation, et la recette dit qui parle dans quel intervalle. Le désaccord d'annotation, qui
est le plancher de bruit habituel de ce genre de mesure, n'existe pas.

Métriques
---------
`vocal_helper.consistency` : dérive par locuteur (sa voix a-t-elle été tenue ensemble),
collision par étiquette (cette étiquette désigne-t-elle une personne), et l'accord à longue
portée en fonction de l'écart — le seul des trois qui réponde à la question du titre.

Usage
-----
    python studies/rediar_speaker_consistency.py --mix /chemin/mix_0000
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import click
import numpy as np

SR = 16000


def reference_from_recipe(recipe: dict) -> list[tuple[float, float, str]]:
    """Les intervalles de vérité terrain, un par tour, étiquetés par locuteur source."""
    return [
        (float(t["mix_start_s"]), float(t["mix_end_s"]), str(t["speaker_id"]))
        for t in recipe.get("turns", [])
        if float(t["mix_end_s"]) > float(t["mix_start_s"])
    ]


async def voiced_spans(pcm: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """Ce que le VAD Silero produirait, c'est-à-dire ce que le direct voit vraiment."""
    import vocal_helper as voh

    inbox: asyncio.Queue = asyncio.Queue()
    outbox: asyncio.Queue = asyncio.Queue()
    frame = int(0.032 * SR)
    for i in range(0, pcm.size, frame):
        inbox.put_nowait({"t0": i / SR, "sample_rate": SR, "pcm": pcm[i : i + frame]})
    inbox.put_nowait(None)
    await voh.SileroVADStage().run(inbox, outbox)
    out = []
    while not outbox.empty():
        item = outbox.get_nowait()
        if item is not None:
            out.append((item["t0"], item["t1"], item["pcm"]))
    return out


def der(reference, hypothesis) -> float:
    """DER au collar 0,25, pour situer la mesure de cohérence à côté du chiffre habituel."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def to_annotation(rows):
        annotation = Annotation()
        for t0, t1, speaker in rows:
            if t1 > t0:
                annotation[Segment(t0, t1)] = speaker
        return annotation

    return float(
        DiarizationErrorRate(collar=0.25)(to_annotation(reference), to_annotation(hypothesis))
    )


@click.command()
@click.option("--mix", "mix_stem", required=True, help="Chemin sans extension (…/mix_0000).")
@click.option("--window-s", default=120.0, show_default=True)
@click.option("--period-s", default=60.0, show_default=True)
@click.option("--gaps", default="30,120,300", show_default=True, help="Écarts à mesurer, en s.")
@click.option(
    "--out", "out_path", default="studies/rediar_speaker_consistency.json", show_default=True
)
def main(mix_stem, window_s, period_s, gaps, out_path):
    """Mesurer dérive, collision et accord à longue portée sur les trois conditions."""
    import audio_helper as ah

    import vocal_helper as voh
    from vocal_helper.consistency import speaker_consistency
    from vocal_helper.rediar import PeriodicOfflineDiarStage, RediarPolicy

    stem = Path(mix_stem)
    recipe = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    audio, rate = ah.load_audio(
        str(stem.with_suffix(".wav")), target_sample_rate=SR, to_mono=True, to_numpy=True
    )
    pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
    assert rate == SR
    reference = reference_from_recipe(recipe)
    duration = pcm.size / SR
    gap_list = [float(g) for g in gaps.split(",") if g.strip()]

    click.echo(
        f"\n── {recipe['mix_id']} · {duration / 60:.1f} min · "
        f"{len(recipe.get('speaker_to_gid', {}))} locuteurs · {len(reference)} tours"
    )
    voiced = asyncio.run(voiced_spans(pcm))
    click.echo(f"   VAD : {len(voiced)} segments voisés")

    conditions: dict[str, list] = {}

    # 1. le différé pleine fenêtre : la référence haute
    started = time.perf_counter()
    conditions["offline_full"] = voh.OfflineDiarStage().diarize(pcm, SR)
    offline_rtf = (time.perf_counter() - started) / duration

    # 2 et 3 : l'en ligne seul, et la re-diarisation, sur le même pilotage
    policy = RediarPolicy(window_s=window_s, period_s=period_s)
    stage = PeriodicOfflineDiarStage(
        online=voh.OnlineDiarStage(), offline=voh.OfflineDiarStage(), policy=policy
    )
    stage._sample_rate = SR
    online_rows: list = []
    started = time.perf_counter()
    for t0, t1, chunk in voiced:
        stage._voiced.append((t0, t1, chunk))
        labelled = stage.online._label({"t0": t0, "t1": t1, "sample_rate": SR, "pcm": chunk})
        if labelled is not None:
            stage._published.append({"t0": t0, "t1": t1, "speaker": labelled["speaker"]})
            online_rows.append((t0, t1, labelled["speaker"]))
        horizon = t1 - (policy.window_s + policy.period_s)
        stage._voiced = [v for v in stage._voiced if v[1] > horizon]
        stage.refresh(now_s=t1, sample_rate=SR)
    if stage._voiced:
        stage.refresh(now_s=stage._voiced[-1][1], sample_rate=SR, force=True)
    rediar_rtf = (time.perf_counter() - started) / duration

    conditions["online_only"] = online_rows
    conditions["rediar_windowed"] = [(r["t0"], r["t1"], r["speaker"]) for r in stage._published]

    rtfs = {"offline_full": offline_rtf, "rediar_windowed": rediar_rtf, "online_only": 0.0}
    rows = []
    click.echo(
        f"\n   {'condition':18} {'DER':>6} {'dérive':>7} {'mal attr.':>10} {'étiq.':>6} "
        + " ".join(f"{'≥' + str(int(g)) + 's':>8}" for g in gap_list)
        + f" {'RTF':>7}"
    )
    for name, hypothesis in conditions.items():
        report = speaker_consistency(reference, hypothesis)
        agreement = {g: report.long_range_agreement(gap_s=g) for g in gap_list}
        row = {
            "condition": name,
            "der": round(der(reference, hypothesis), 4),
            "worst_drift": round(report.worst_drift, 4),
            "worst_collision": round(report.worst_collision, 4),
            "misattributed_share": round(report.misattributed_share, 4),
            "labels": len(report.collision),
            "long_range": {str(int(g)): round(v, 4) for g, v in agreement.items()},
            "rtf": round(rtfs[name], 4),
        }
        rows.append(row)
        click.echo(
            f"   {name:18} {row['der']:6.3f} {row['worst_drift']:7.3f} "
            f"{row['misattributed_share']:10.3f} {row['labels']:6d} "
            + " ".join(f"{agreement[g]:8.3f}" for g in gap_list)
            + f" {row['rtf']:7.3f}"
        )

    click.echo(f"\n   refreshes : {stage.refreshes} · révisions : {len(stage.revisions)}")
    click.echo(f"   locuteurs découverts : {sorted(stage._references)}")

    Path(out_path).write_text(
        json.dumps(
            {
                "mix": recipe["mix_id"],
                "duration_s": round(duration, 1),
                "true_speakers": len(recipe.get("speaker_to_gid", {})),
                "window_s": window_s,
                "period_s": period_s,
                "refreshes": stage.refreshes,
                "revisions": len(stage.revisions),
                "conditions": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    click.echo(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
