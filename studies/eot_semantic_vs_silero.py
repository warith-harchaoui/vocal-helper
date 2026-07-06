"""Semantic EOT (LiveKit-style) vs Silero-only — false-cut benchmark.

Question
--------
LiveKit's blog (April 2026, "Solving end-of-turn detection") reports
**9.9 %** false-cutoff rate at 300 ms semantic latency, **4.5 %** at
600 ms, and a **39 %** reduction in false-positive interruptions when
their distilled Qwen2.5-0.5B EOT model is fused with Silero VAD. Does
adding :class:`SemanticEOTStage` to vocal-helper produce a measurable
improvement on AMI ?

We benchmark **false-cut rate** : the share of voiced segments that
the VAD closes mid-sentence (i.e. before the speaker's punctuation
mark in the reference transcript). A perfect detector cuts only at
sentence ends.

Protocol
--------
- corpus  : AMI dev-slice IS1008a (16 min, 4 speakers).
- baseline: ``SileroVADStage`` only — every closed run is one segment.
- candidate : ``SileroVADStage`` → ``SemanticEOTStage``.
- per closed run :
    1. Whisper-transcribe the segment (large-v3-turbo-q5_0, locked
       to ``en``).
    2. Inspect the trailing token : if it ends in ``.``, ``?`` or
       ``!``, the cut is *clean* ; otherwise *false*.
- metric : false-cut rate = #false_cuts / #segments.
- timing : per-segment wall time for EOT classification — we report
  median classifier latency, since LiveKit advertises ≤ 25 ms.

Caveats
-------
Punctuation-as-EOT proxy is imperfect — speakers sometimes pause at
clauses without punctuation, and Whisper occasionally inserts
spurious commas. We use it as a rough leading indicator ; the real
production metric ("user feels not interrupted") needs a human eval
which is out of scope.

Author : Warith HARCHAOUI — 2026-06-30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from vocal_helper.eot import SemanticEOTStage
from vocal_helper.types import PcmFrame, VoicedSegment
from vocal_helper.vad import SileroVADStage

AMI_ROOT = Path(
    "/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice"
)
DEFAULT_MEETING = "IS1008a"
SR = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SR * FRAME_MS // 1000
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/"
    "vocal_helper_eot_2026-06-30.log"
)
PUNCT_TERMINAL = (".", "?", "!")


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(DEFAULT_LOG, "a") as f:
        f.write(msg + "\n")


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


async def feed_audio(audio: np.ndarray, queue: asyncio.Queue) -> None:
    """Push the audio as PcmFrames through ``queue``, then a ``None`` sentinel."""
    cursor = 0
    while cursor < audio.shape[0]:
        block = audio[cursor : cursor + FRAME_SAMPLES]
        if block.shape[0] < FRAME_SAMPLES:
            block = np.concatenate(
                [block, np.zeros(FRAME_SAMPLES - block.shape[0], dtype=np.float32)],
                axis=0,
            )
        await queue.put(PcmFrame(
            t0=cursor / float(SR), sample_rate=SR, pcm=block,
        ))
        cursor += FRAME_SAMPLES
    await queue.put(None)


async def collect_segments(queue: asyncio.Queue) -> list[VoicedSegment]:
    out: list[VoicedSegment] = []
    while True:
        item = await queue.get()
        if item is None:
            return out
        out.append(item)


def transcribe(pcm: np.ndarray, sr: int, whisper_model) -> str:
    try:
        segs = whisper_model.transcribe(pcm)
    except Exception:  # noqa: BLE001
        return ""
    return " ".join((s.text or "").strip() for s in segs).strip()


def is_false_cut(text: str) -> bool:
    """A cut is *false* if the segment does NOT end with a terminal punct."""
    stripped = text.rstrip()
    if not stripped:
        return False  # empty — neither clean nor false
    return stripped[-1] not in PUNCT_TERMINAL


async def baseline_run(audio: np.ndarray) -> list[VoicedSegment]:
    """Silero VAD alone — collect every emitted segment."""
    vad = SileroVADStage()
    inbox: asyncio.Queue = asyncio.Queue()
    outbox: asyncio.Queue = asyncio.Queue()
    feeder = asyncio.create_task(feed_audio(audio, inbox))
    vad_task = asyncio.create_task(vad.run(inbox, outbox))
    segs = await collect_segments(outbox)
    await feeder
    await vad_task
    return segs


async def eot_run(audio: np.ndarray) -> tuple[list[VoicedSegment], list[float]]:
    """VAD → SemanticEOTStage cascade, with per-segment timing."""
    vad = SileroVADStage()
    eot = SemanticEOTStage()
    q1: asyncio.Queue = asyncio.Queue()
    q2: asyncio.Queue = asyncio.Queue()
    q3: asyncio.Queue = asyncio.Queue()
    walls: list[float] = []

    # Wrap EOT.run to record per-call classification latency.
    original = eot._classify

    def _timed_classify(text: str) -> bool:
        t0 = time.perf_counter()
        out = original(text)
        walls.append(time.perf_counter() - t0)
        return out

    eot._classify = _timed_classify

    feeder = asyncio.create_task(feed_audio(audio, q1))
    vad_task = asyncio.create_task(vad.run(q1, q2))
    eot_task = asyncio.create_task(eot.run(q2, q3))
    segs = await collect_segments(q3)
    await feeder
    await vad_task
    await eot_task
    return segs, walls


def evaluate(segs: list[VoicedSegment], whisper_model, label: str) -> dict:
    n_total = len(segs)
    n_false = 0
    n_empty = 0
    for s in segs:
        text = transcribe(s["pcm"], s["sample_rate"], whisper_model)
        if not text:
            n_empty += 1
        elif is_false_cut(text):
            n_false += 1
    rate = n_false / max(1, n_total - n_empty)
    log(
        f"  {label:<14s}  n_segs={n_total:>3d}  n_empty={n_empty:>2d}  "
        f"n_false_cuts={n_false:>3d}  false_cut_rate={rate:.3f}"
    )
    return {
        "label": label,
        "n_segs": n_total,
        "n_empty": n_empty,
        "n_false": n_false,
        "false_cut_rate": rate,
    }


async def amain(args: argparse.Namespace) -> None:
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")
    log("# Semantic EOT vs Silero-only — 2026-06-30")
    log(f"# meeting : {args.meeting}")

    wav = AMI_ROOT / args.meeting / "mix.wav"
    audio, sr = read_mono_wav(wav)
    log(f"# duration : {audio.shape[0] / SR:.0f}s")

    # Whisper model used as the EOT *evaluator* (separate from
    # SemanticEOTStage's internal whisper instance which is used for
    # the classifier's partial transcript).
    from pywhispercpp.model import Model

    log("\n[setup] loading evaluator whisper.cpp …")
    whisper = Model(
        "large-v3-turbo-q5_0",
        n_threads=6,
        language="en",
        print_realtime=False,
        print_progress=False,
    )

    log("\n=== baseline : Silero VAD only ===")
    base_segs = await baseline_run(audio)
    base_res = evaluate(base_segs, whisper, "silero")

    log("\n=== candidate : Silero + SemanticEOTStage ===")
    eot_segs, eot_walls = await eot_run(audio)
    eot_res = evaluate(eot_segs, whisper, "silero+eot")

    if eot_walls:
        log(
            f"\n  EOT classifier latency : median={statistics.median(eot_walls)*1000:.0f} ms  "
            f"min={min(eot_walls)*1000:.0f} ms  max={max(eot_walls)*1000:.0f} ms  "
            f"(n_calls={len(eot_walls)})"
        )

    delta = eot_res["false_cut_rate"] - base_res["false_cut_rate"]
    log(f"\nfalse_cut_rate delta : {delta:+.3f} (negative = EOT helps)")
    if delta < -0.05:
        log("Recommendation : enable SemanticEOTStage by default in vocal-helper.")
    elif delta < 0:
        log("Recommendation : marginal improvement ; keep EOT opt-in.")
    else:
        log("Recommendation : EOT does not help on this corpus ; keep opt-in.")

    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(json.dumps({
        "meeting": args.meeting,
        "baseline": base_res,
        "candidate": eot_res,
        "eot_classifier_walls_s": eot_walls,
    }, indent=2))
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meeting", default=DEFAULT_MEETING)
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
