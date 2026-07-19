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

AMI_ROOT = Path("/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice")
DEFAULT_MEETING = "IS1008a"
SR = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SR * FRAME_MS // 1000
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/vocal_helper_eot_2026-06-30.log"
)
PUNCT_TERMINAL = (".", "?", "!")


def log(msg: str) -> None:
    """Echo a study line to stdout and append it to the on-disk log.

    Parameters
    ----------
    msg : str
        Line to emit. Printed live and mirrored to :data:`DEFAULT_LOG`
        so the full run survives after the terminal scrolls away.

    Returns
    -------
    None
    """
    # stdout is the live view; flush keeps ordering deterministic.
    print(msg, flush=True)
    # Append so successive calls build one durable transcript of the run.
    with open(DEFAULT_LOG, "a") as f:
        f.write(msg + "\n")


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as a mono float32 waveform.

    Parameters
    ----------
    path : Path
        Path to the WAV file to load.

    Returns
    -------
    tuple[numpy.ndarray, int]
        The mono ``float32`` waveform and its native sample rate.

    Notes
    -----
    The VAD / EOT cascade is single-channel, so multi-channel input is
    collapsed by averaging across channels.
    """
    # Decode straight to float32 to avoid a later int→float conversion.
    audio, sr = sf.read(str(path), dtype="float32")
    # Fold any multi-channel AMI mix down to mono by channel-averaging.
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


async def feed_audio(audio: np.ndarray, queue: asyncio.Queue) -> None:
    """Stream a waveform into ``queue`` as fixed-size PCM frames.

    Slices the waveform into :data:`FRAME_SAMPLES`-long blocks, wraps
    each in a :class:`PcmFrame`, enqueues them in order, then pushes a
    ``None`` sentinel to signal end-of-stream to the consumer stage.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono ``float32`` waveform to stream.
    queue : asyncio.Queue
        Destination queue feeding the first cascade stage.

    Returns
    -------
    None
    """
    # Walk the waveform in fixed strides, one frame per iteration.
    cursor = 0
    while cursor < audio.shape[0]:
        # Take the next FRAME_SAMPLES window (may be short at the tail).
        block = audio[cursor : cursor + FRAME_SAMPLES]
        # The final block is usually short; zero-pad it to a full frame
        # so downstream stages always see uniform frame lengths.
        if block.shape[0] < FRAME_SAMPLES:
            block = np.concatenate(
                [block, np.zeros(FRAME_SAMPLES - block.shape[0], dtype=np.float32)],
                axis=0,
            )
        # Timestamp each frame by its start sample so segment times are real.
        await queue.put(
            PcmFrame(
                t0=cursor / float(SR),
                sample_rate=SR,
                pcm=block,
            )
        )
        cursor += FRAME_SAMPLES
    # Sentinel: tells the consumer no more frames are coming.
    await queue.put(None)


async def collect_segments(queue: asyncio.Queue) -> list[VoicedSegment]:
    """Drain a queue of voiced segments until the end-of-stream sentinel.

    Parameters
    ----------
    queue : asyncio.Queue
        Output queue of a cascade stage. Items are :class:`VoicedSegment`
        objects terminated by a single ``None`` sentinel.

    Returns
    -------
    list[VoicedSegment]
        Every segment received before the sentinel, in arrival order.
    """
    out: list[VoicedSegment] = []
    # Pull segments until the producer signals it is done.
    while True:
        item = await queue.get()
        # ``None`` is the end-of-stream marker pushed by the producer.
        if item is None:
            return out
        # Real segment — accumulate and keep draining.
        out.append(item)


def transcribe(pcm: np.ndarray, sr: int, whisper_model) -> str:
    """Transcribe one PCM segment to text, swallowing decode errors.

    Parameters
    ----------
    pcm : numpy.ndarray
        Mono ``float32`` waveform of a single voiced segment.
    sr : int
        Sample rate of ``pcm`` in Hz (kept for signature symmetry; the
        whisper model is already configured for :data:`SR`).
    whisper_model : object
        A loaded ``pywhispercpp`` model exposing ``.transcribe(pcm)``.

    Returns
    -------
    str
        Space-joined, stripped transcript, or ``""`` if decoding failed.

    Notes
    -----
    Failures return an empty string on purpose: the caller treats empty
    transcripts as "neither clean nor false" so a flaky decode never
    inflates the false-cut count.
    """
    try:
        segs = whisper_model.transcribe(pcm)
    except Exception:  # noqa: BLE001
        # A single bad segment must not abort the whole evaluation pass.
        return ""
    # Concatenate the whisper sub-segments into one transcript string.
    return " ".join((s.text or "").strip() for s in segs).strip()


def is_false_cut(text: str) -> bool:
    """Classify a segment cut as *false* (mid-sentence) via punctuation.

    Parameters
    ----------
    text : str
        Transcript of the closed voiced segment.

    Returns
    -------
    bool
        ``True`` if the segment does NOT end with terminal punctuation
        (``.``/``?``/``!``) — our proxy for a mid-sentence cut. ``False``
        for a clean cut, and also ``False`` for an empty transcript.
    """
    stripped = text.rstrip()
    # Empty transcript carries no punctuation signal — exclude it from
    # both the clean and false tallies.
    if not stripped:
        return False  # empty — neither clean nor false
    # Terminal punctuation ⇒ clean cut; anything else ⇒ false cut.
    return stripped[-1] not in PUNCT_TERMINAL


async def baseline_run(audio: np.ndarray) -> list[VoicedSegment]:
    """Run the Silero-VAD-only baseline and collect its segments.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono ``float32`` waveform of the whole meeting.

    Returns
    -------
    list[VoicedSegment]
        Every voiced segment Silero emits — the baseline hypothesis.
    """
    # Fresh VAD instance so the baseline shares no state with the candidate.
    vad = SileroVADStage()
    # Two queues wire feeder → VAD → collector as an async pipeline.
    inbox: asyncio.Queue = asyncio.Queue()
    outbox: asyncio.Queue = asyncio.Queue()
    # Launch producer (frame feeder) and the VAD stage concurrently.
    feeder = asyncio.create_task(feed_audio(audio, inbox))
    vad_task = asyncio.create_task(vad.run(inbox, outbox))
    # Consume the VAD output on this coroutine until the sentinel.
    segs = await collect_segments(outbox)
    # Join the background tasks so exceptions surface deterministically.
    await feeder
    await vad_task
    return segs


async def eot_run(audio: np.ndarray) -> tuple[list[VoicedSegment], list[float]]:
    """Run the VAD → SemanticEOT cascade and time each classifier call.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono ``float32`` waveform of the whole meeting.

    Returns
    -------
    tuple[list[VoicedSegment], list[float]]
        The candidate hypothesis segments and the list of per-call EOT
        classifier latencies in seconds.
    """
    # Fresh stage instances per run so no state leaks from the baseline.
    vad = SileroVADStage()
    eot = SemanticEOTStage()
    # Three queues chain feeder → VAD → EOT → collector.
    q1: asyncio.Queue = asyncio.Queue()
    q2: asyncio.Queue = asyncio.Queue()
    q3: asyncio.Queue = asyncio.Queue()
    walls: list[float] = []

    # Wrap EOT.run to record per-call classification latency.
    # Keep a handle to the real classifier so the wrapper can delegate.
    original = eot._classify

    def _timed_classify(text: str) -> bool:
        """Time-instrumented proxy for ``SemanticEOTStage._classify``.

        Parameters
        ----------
        text : str
            Partial transcript passed to the real EOT classifier.

        Returns
        -------
        bool
            The classifier's end-of-turn decision, unchanged.

        Notes
        -----
        Appends the wall-clock latency of the wrapped call to the
        enclosing ``walls`` list so the study can report median EOT
        latency against LiveKit's ≤ 25 ms claim.
        """
        # Time only the wrapped classifier call, nothing around it.
        t0 = time.perf_counter()
        out = original(text)
        walls.append(time.perf_counter() - t0)
        return out

    # Monkey-patch the instance method so timing is transparent to the stage.
    eot._classify = _timed_classify

    # Launch the full cascade; each stage forwards through its queue.
    feeder = asyncio.create_task(feed_audio(audio, q1))
    vad_task = asyncio.create_task(vad.run(q1, q2))
    eot_task = asyncio.create_task(eot.run(q2, q3))
    # Drain the final stage's output on this coroutine.
    segs = await collect_segments(q3)
    # Join every background task so nothing is left dangling.
    await feeder
    await vad_task
    await eot_task
    return segs, walls


def evaluate(segs: list[VoicedSegment], whisper_model, label: str) -> dict:
    """Score one hypothesis by its false-cut rate and log the breakdown.

    Parameters
    ----------
    segs : list[VoicedSegment]
        Hypothesis segments produced by a run (baseline or candidate).
    whisper_model : object
        Loaded whisper model used to transcribe each segment for the
        punctuation-based false-cut check.
    label : str
        Short human-readable name for this run (e.g. ``"silero"``),
        used in the logged result line and the returned record.

    Returns
    -------
    dict
        Result record with keys ``label``, ``n_segs``, ``n_empty``,
        ``n_false`` and ``false_cut_rate``.

    Notes
    -----
    Empty transcripts are excluded from the denominator so undecodable
    segments neither help nor hurt the measured false-cut rate.
    """
    # Tally counters over the run's segments.
    n_total = len(segs)
    n_false = 0
    n_empty = 0
    # Walk every closed segment and bucket it as empty / false / clean.
    for s in segs:
        # Transcribe each closed segment; empty means the decode gave nothing.
        text = transcribe(s["pcm"], s["sample_rate"], whisper_model)
        if not text:
            n_empty += 1
        # A segment not ending in terminal punctuation is a false cut.
        elif is_false_cut(text):
            n_false += 1
    # Rate is over *decodable* segments only; max(1, …) avoids /0.
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
    """Run the async EOT-vs-Silero benchmark end to end.

    Loads one AMI meeting, runs the Silero-only baseline and the
    Silero+SemanticEOT candidate, scores both by false-cut rate, reports
    the delta plus classifier latency, and persists a JSON sidecar.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments; uses ``args.meeting``.

    Returns
    -------
    None
    """
    # Reset the log file, then write the run header.
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")
    log("# Semantic EOT vs Silero-only — 2026-06-30")
    log(f"# meeting : {args.meeting}")

    # Load the mixed meeting audio once; both runs replay the same signal.
    wav = AMI_ROOT / args.meeting / "mix.wav"
    audio, sr = read_mono_wav(wav)
    # Log duration so RTF-style timings later have context.
    log(f"# duration : {audio.shape[0] / SR:.0f}s")

    # Whisper model used as the EOT *evaluator* (separate from
    # SemanticEOTStage's internal whisper instance which is used for
    # the classifier's partial transcript).
    from pywhispercpp.model import Model

    # Lock the evaluator to English — AMI is English, and locking skips
    # per-segment LID so scoring stays fast and deterministic.
    log("\n[setup] loading evaluator whisper.cpp …")
    whisper = Model(
        "large-v3-turbo-q5_0",
        n_threads=6,
        language="en",
        print_realtime=False,
        print_progress=False,
    )

    # Baseline: Silero VAD alone, one segment per closed run.
    log("\n=== baseline : Silero VAD only ===")
    base_segs = await baseline_run(audio)
    # Score the baseline hypothesis to get its false-cut rate.
    base_res = evaluate(base_segs, whisper, "silero")

    # Candidate: Silero VAD followed by the semantic EOT stage.
    log("\n=== candidate : Silero + SemanticEOTStage ===")
    eot_segs, eot_walls = await eot_run(audio)
    # Score the candidate hypothesis; eot_walls carries classifier latencies.
    eot_res = evaluate(eot_segs, whisper, "silero+eot")

    # Report classifier latency against LiveKit's ≤ 25 ms advertised bar.
    # Guard on non-empty walls: no EOT calls means nothing to summarize.
    if eot_walls:
        log(
            f"\n  EOT classifier latency : median={statistics.median(eot_walls) * 1000:.0f} ms  "
            f"min={min(eot_walls) * 1000:.0f} ms  max={max(eot_walls) * 1000:.0f} ms  "
            f"(n_calls={len(eot_walls)})"
        )

    # Compare candidate against baseline on the headline metric.
    # Negative delta = the EOT stage lowered the false-cut rate (good).
    delta = eot_res["false_cut_rate"] - base_res["false_cut_rate"]
    log(f"\nfalse_cut_rate delta : {delta:+.3f} (negative = EOT helps)")
    # Decision thresholds: a >5 pp drop justifies enabling EOT by default;
    # any smaller improvement keeps it opt-in; no improvement rejects it.
    if delta < -0.05:
        log("Recommendation : enable SemanticEOTStage by default in vocal-helper.")
    elif delta < 0:
        log("Recommendation : marginal improvement ; keep EOT opt-in.")
    else:
        log("Recommendation : EOT does not help on this corpus ; keep opt-in.")

    # Persist the full comparison (incl. raw latencies) as a JSON sidecar.
    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "meeting": args.meeting,
                "baseline": base_res,
                "candidate": eot_res,
                "eot_classifier_walls_s": eot_walls,
            },
            indent=2,
        )
    )
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


def main() -> None:
    """Parse CLI arguments and drive the async benchmark.

    Thin synchronous entry point: it exists because :func:`amain` is a
    coroutine (the cascade is async) and needs an event loop to run.

    Returns
    -------
    None
    """
    # Only knob is which meeting to benchmark.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meeting", default=DEFAULT_MEETING)
    args = p.parse_args()
    # Bridge sync → async: spin up an event loop for the coroutine.
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
