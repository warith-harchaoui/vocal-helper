"""Le DER réel de la re-diarisation en direct, contre le différé et contre l'en ligne.

Question
--------
`vocal_helper.rediar` promet la qualité du différé sur un flux. Le chiffre annoncé
(DER 0,122) vient pourtant d'une passe différée **pleine fenêtre**, alors que la version
bornée ne voit que `window_s` secondes à la fois, plus une amorce. Moins de contexte veut
dire au moins un peu plus d'erreur : le vrai nombre est quelque part entre 0,122 et le
0,586 de l'étage en ligne, et tant qu'il n'est pas mesuré, la promesse n'est pas tenue.

Protocole
---------
Trois conditions sur le même audio et la même vérité terrain, sur AMI dev-slice :

1. **différé pleine fenêtre** — la référence haute, ce que le différé sait faire ;
2. **re-diarisation bornée** — les vrais étages pilotés comme en direct : source → VAD
   Silero → `PeriodicOfflineDiarStage`. La transcription est volontairement absente : elle
   ne change aucune étiquette de locuteur et coûterait le gros du temps de calcul ;
3. **en ligne seul** — le plancher bas, ce que coûte la latence.

DER par `pyannote.metrics`, collar 0,25, comme les autres campagnes de ce dépôt. Le RTF
mesuré au passage, parce qu'une qualité qui ne tient pas sous 1 ne compte pas.

Usage
-----
    python studies/rediar_live_der.py --meetings IS1008a ES2011a
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import click
import numpy as np

AMI_ROOT = Path("/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice")
SR = 16000


def load_reference(rttm: Path):
    """La vérité terrain d'une réunion, en Annotation pyannote."""
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for line in rttm.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
        if duration > 0:
            annotation[Segment(start, start + duration)] = speaker
    return annotation


def to_annotation(segments):
    """``[(t0, t1, speaker)]`` en Annotation pyannote."""
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for t0, t1, speaker in segments:
        if t1 > t0:
            annotation[Segment(t0, t1)] = speaker
    return annotation


def score(reference, hypothesis) -> float:
    """DER avec le collar de 0,25 s des autres campagnes d'ici."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    return float(metric(reference, hypothesis))


async def voiced_spans(pcm: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """Les segments voisés que le VAD Silero produirait sur ce flux.

    On passe par le vrai étage : simuler la détection d'activité vocale depuis la vérité
    terrain reviendrait à donner la réponse au système qu'on mesure.
    """
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


def run_offline(pcm: np.ndarray) -> tuple[list, float]:
    """Condition 1 : le différé pleine fenêtre."""
    import vocal_helper as voh

    stage = voh.OfflineDiarStage(backend="pyannote")
    started = time.perf_counter()
    segments = stage.diarize(pcm, SR)
    return segments, time.perf_counter() - started


def run_rediar(voiced, policy) -> tuple[list, float, int]:
    """Condition 2 : la re-diarisation bornée, pilotée comme en direct."""
    import vocal_helper as voh
    from vocal_helper.rediar import PeriodicOfflineDiarStage

    stage = PeriodicOfflineDiarStage(
        online=voh.OnlineDiarStage(backend="pyannote"),
        offline=voh.OfflineDiarStage(backend=policy.backend),
        policy=policy,
    )
    started = time.perf_counter()
    stage._sample_rate = SR
    for t0, t1, chunk in voiced:
        stage._voiced.append((t0, t1, chunk))
        labelled = stage.online._label({"t0": t0, "t1": t1, "sample_rate": SR, "pcm": chunk})
        if labelled is not None:
            stage._published.append({"t0": t0, "t1": t1, "speaker": labelled["speaker"]})
        horizon = t1 - (policy.window_s + policy.period_s)
        stage._voiced = [v for v in stage._voiced if v[1] > horizon]
        stage.refresh(now_s=t1, sample_rate=SR)
    if stage._voiced:
        stage.refresh(now_s=stage._voiced[-1][1], sample_rate=SR, force=True)
    elapsed = time.perf_counter() - started
    segments = [(r["t0"], r["t1"], r["speaker"]) for r in stage._published]
    return segments, elapsed, stage.refreshes


def run_online(voiced) -> tuple[list, float]:
    """Condition 3 : l'étage en ligne seul."""
    import vocal_helper as voh

    stage = voh.OnlineDiarStage(backend="pyannote")
    started = time.perf_counter()
    out = []
    for t0, t1, chunk in voiced:
        labelled = stage._label({"t0": t0, "t1": t1, "sample_rate": SR, "pcm": chunk})
        if labelled is not None:
            out.append((t0, t1, labelled["speaker"]))
    return out, time.perf_counter() - started


@click.command()
@click.option("--meetings", multiple=True, default=("IS1008a",), help="Réunions AMI à mesurer.")
@click.option("--window-s", default=180.0, show_default=True)
@click.option("--period-s", default=60.0, show_default=True)
@click.option("--out", "out_path", default="studies/rediar_live_der.json", show_default=True)
def main(meetings, window_s, period_s, out_path):
    """Mesurer les trois conditions et écrire le relevé."""
    import audio_helper as ah

    from vocal_helper.rediar import RediarPolicy

    policy = RediarPolicy(window_s=window_s, period_s=period_s)
    results = []

    for name in meetings:
        folder = AMI_ROOT / name
        wav, rttm = folder / "mix.wav", folder / "words.rttm"
        if not (wav.is_file() and rttm.is_file()):
            click.echo(f"  {name} : absent, ignoré")
            continue

        # audio-helper returns (audio, rate) and takes ``target_sample_rate``; it also
        # hands back a torch tensor unless asked for numpy. Both are easy to get wrong
        # from memory, which is why this was checked against the real signature.
        audio, rate = ah.load_audio(str(wav), target_sample_rate=SR, to_mono=True, to_numpy=True)
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        assert rate == SR, f"{name}: attendu {SR} Hz, reçu {rate}"
        duration = pcm.size / SR
        reference = load_reference(rttm)
        click.echo(f"\n── {name} · {duration / 60:.1f} min")

        voiced = asyncio.run(voiced_spans(pcm))
        click.echo(f"   VAD : {len(voiced)} segments voisés")

        row = {"meeting": name, "duration_s": round(duration, 1), "voiced": len(voiced)}

        # Les lambdas lient leurs arguments par défaut : sans ça elles captureraient
        # la variable de boucle et les trois conditions mesureraient la dernière réunion.
        for label, runner in (
            ("offline_full", lambda pcm=pcm: run_offline(pcm)),
            ("rediar_windowed", lambda v=voiced: run_rediar(v, policy)),
            ("online_only", lambda v=voiced: run_online(v)),
        ):
            try:
                outcome = runner()
                segments, elapsed = outcome[0], outcome[1]
                der = score(reference, to_annotation(segments))
                row[label] = {
                    "der": round(der, 4),
                    "rtf": round(elapsed / duration, 4),
                    "segments": len(segments),
                }
                if len(outcome) > 2:
                    row[label]["refreshes"] = outcome[2]
                click.echo(
                    f"   {label:16} DER {der:.3f}  RTF {elapsed / duration:.3f}"
                    f"  ({len(segments)} segments)"
                )
            except Exception as exc:  # noqa: BLE001 - une condition qui casse ne doit pas tuer la campagne
                row[label] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                click.echo(f"   {label:16} ERREUR {row[label]['error']}")

        results.append(row)

    Path(out_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
