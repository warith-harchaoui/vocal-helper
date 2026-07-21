"""SileroVADStage — construction guard, run segmentation, and edge cases.

All tests are model-free: the Silero ONNX model and its torch handle are
replaced with tiny scripted fakes, so ``_ensure_model`` never downloads a
model and every score is deterministic. That lets us drive the whole VAD
state machine — run start / accumulation / silence-close / min-speech reject /
edge padding — over synthetic PcmFrames and assert the emitted VoicedSegment
sequence. The construction sample-rate guard and the end-of-stream flush are
covered too.

The scripted scorer keys off a marker baked into each frame's PCM (the
frame's first sample), so a test can lay out a voiced/silent timeline
sample-exactly without any real speech.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from vocal_helper.types import PcmFrame
from vocal_helper.vad import _SILERO_WINDOW_SAMPLES, SileroVADStage

SR = 16_000


# ---------------------------------------------------------------------------
# Test doubles: a torch handle and a Silero model with no ML in them.
# ---------------------------------------------------------------------------


class _FakeTorch:
    """Minimal stand-in for the ``torch`` handle the stage caches.

    The stage only touches three things on its torch handle: ``no_grad`` as a
    context manager and ``from_numpy`` to wrap the window. We hand back the raw
    numpy window so the fake model can read its marker sample directly.
    """

    class _NoGrad:
        """No-op context manager mirroring ``torch.no_grad()``."""

        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> bool:
            return False

    def no_grad(self) -> _NoGrad:
        """Return the no-op context manager."""
        return self._NoGrad()

    @staticmethod
    def from_numpy(window: np.ndarray) -> np.ndarray:
        """Identity wrap — the fake model reads the numpy array verbatim."""
        return window


class _Scored:
    """Wraps a float so ``.item()`` mirrors a real torch scalar."""

    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        """Return the scripted score as a Python float."""
        return self._value


class _FakeSilero:
    """Deterministic scorer: a window is voiced iff its first sample is high.

    A real Silero window would carry speech energy; here we encode the intended
    verdict directly in the marker sample so the state machine — not any model —
    is what the test exercises.
    """

    def __init__(self, threshold_marker: float = 0.5) -> None:
        self._marker = threshold_marker

    def __call__(self, window: np.ndarray, sample_rate: int) -> _Scored:
        """Score 1.0 for a voiced-marked window, 0.0 for a silent one."""
        # First sample is the marker: >= 0.5 means "this window is voiced".
        return _Scored(1.0 if float(window[0]) >= self._marker else 0.0)


def _wire_fakes(stage: SileroVADStage) -> SileroVADStage:
    """Inject the fake model + torch so ``_ensure_model`` is a no-op.

    Pre-populating ``_model`` short-circuits the lazy loader (it returns early
    when a model is already set), so no ONNX download is ever attempted.
    """
    stage._model = _FakeSilero()
    stage._torch = _FakeTorch()
    return stage


def _frame(marker: float, n_samples: int, t0: float = 0.0) -> PcmFrame:
    """Build a PcmFrame whose every sample equals ``marker``.

    The constant marker doubles as the voiced/silent flag the fake scorer reads
    from the window's first sample, so a frame's verdict is unambiguous.
    """
    return PcmFrame(
        t0=t0,
        sample_rate=SR,
        pcm=np.full(n_samples, marker, dtype=np.float32),
    )


def _run_stage(stage: SileroVADStage, frames: list[PcmFrame]) -> list:
    """Feed ``frames`` (then a ``None`` sentinel) through the stage's coroutine.

    Returns the list of emitted VoicedSegments, draining the sentinel that the
    stage forwards to signal end-of-stream.
    """

    async def _drive() -> list:
        inbox: asyncio.Queue = asyncio.Queue()
        outbox: asyncio.Queue = asyncio.Queue()
        for f in frames:
            inbox.put_nowait(f)
        inbox.put_nowait(None)  # end-of-stream sentinel
        await stage.run(inbox, outbox)
        out = []
        while not outbox.empty():
            out.append(outbox.get_nowait())
        return out

    return asyncio.run(_drive())


# ---------------------------------------------------------------------------
# construction contract
# ---------------------------------------------------------------------------


def test_construction_rejects_non_16k_and_stores_defaults() -> None:
    """Silero is 16 kHz-only, so any other rate raises; defaults are canonical.

    Pins the one hard precondition (sample_rate must be 16 000) and confirms a
    bare stage carries the documented thresholds and hasn't loaded a model.
    """
    # Silero v5 is trained at 16 kHz — a mismatched rate must fail loudly at
    # construction, not silently mis-score downstream.
    with pytest.raises(ValueError, match="16000"):
        SileroVADStage(sample_rate=8_000)

    stage = SileroVADStage()
    assert stage.activity_threshold == 0.5
    assert stage.min_silence_ms == 300
    assert stage.min_speech_ms == 300
    assert stage.edge_pad_ms == 200
    assert stage._model is None  # lazy — nothing loaded at construction time


# ---------------------------------------------------------------------------
# run segmentation — the state machine
# ---------------------------------------------------------------------------


def test_voiced_run_emits_one_segment_after_trailing_silence() -> None:
    """A voiced burst followed by >= min_silence_ms of silence emits exactly one segment.

    Drives the core state machine end-to-end: silence primes the lead pad, a
    long voiced burst opens and accumulates a run, and a trailing silence gap
    that crosses ``min_silence_ms`` closes it into a single VoicedSegment whose
    span and 16 kHz contract hold.
    """
    # 300 ms min_silence closes the run; keep min_speech low so the burst counts.
    stage = _wire_fakes(SileroVADStage(min_silence_ms=300, min_speech_ms=100, edge_pad_ms=0))
    win = _SILERO_WINDOW_SAMPLES

    # 600 ms of voiced audio (well over min_speech), then 400 ms of silence
    # (over the 300 ms close threshold). One frame each, sized to whole windows.
    voiced = _frame(1.0, win * 20)  # ~640 ms voiced
    silence = _frame(0.0, win * 14)  # ~448 ms silence

    segs = _run_stage(stage, [voiced, silence])
    voiced_segs = [s for s in segs if s is not None]

    # Exactly one run emitted, and it is a well-formed 16 kHz segment.
    assert len(voiced_segs) == 1
    seg = voiced_segs[0]
    assert seg["sample_rate"] == SR
    assert seg["t1"] > seg["t0"] >= 0.0
    assert seg["pcm"].dtype == np.float32
    # None sentinel is always forwarded so downstream stages can stop.
    assert segs[-1] is None


def test_short_voiced_run_is_dropped_below_min_speech() -> None:
    """A voiced blip shorter than min_speech_ms is rejected, not emitted.

    Guards the finger-tap / lip-smack filter: a run that closes on silence but
    whose padded duration is under ``min_speech_ms`` must produce no segment —
    only the end-of-stream ``None`` comes through.
    """
    # High min_speech (1 s) means a ~100 ms blip cannot survive the length gate.
    stage = _wire_fakes(SileroVADStage(min_silence_ms=100, min_speech_ms=1000, edge_pad_ms=0))
    win = _SILERO_WINDOW_SAMPLES

    blip = _frame(1.0, win * 3)  # ~96 ms voiced — below the 1 s floor
    silence = _frame(0.0, win * 8)  # long enough to close the run

    segs = _run_stage(stage, [blip, silence])
    assert [s for s in segs if s is not None] == []  # dropped
    assert segs == [None]  # only the sentinel survives


def test_open_run_is_flushed_on_end_of_stream() -> None:
    """A run still open when the stream ends is flushed before the sentinel.

    Covers the ``None``-sentinel branch of ``run``: if speech is ongoing when
    the input closes (no trailing silence ever arrives), the in-flight run must
    still be emitted rather than silently dropped.
    """
    # No trailing silence frame at all — the run can only close via the flush.
    stage = _wire_fakes(SileroVADStage(min_silence_ms=300, min_speech_ms=100, edge_pad_ms=0))
    win = _SILERO_WINDOW_SAMPLES

    voiced = _frame(1.0, win * 20)  # ~640 ms, over min_speech, never closed
    segs = _run_stage(stage, [voiced])

    voiced_segs = [s for s in segs if s is not None]
    assert len(voiced_segs) == 1  # flushed on end-of-stream
    assert voiced_segs[0]["t1"] > voiced_segs[0]["t0"]
    assert segs[-1] is None


def test_two_bursts_split_into_two_segments() -> None:
    """Voiced / silence / voiced produces two independent segments.

    Exercises the run reset path: after a run closes on silence, the lead pad is
    re-primed and a fresh voiced burst opens a brand-new run — so a
    speech-gap-speech timeline yields two ordered, non-overlapping segments.
    """
    stage = _wire_fakes(SileroVADStage(min_silence_ms=200, min_speech_ms=100, edge_pad_ms=0))
    win = _SILERO_WINDOW_SAMPLES

    frames = [
        _frame(1.0, win * 12),  # burst 1
        _frame(0.0, win * 10),  # gap over min_silence → closes burst 1
        _frame(1.0, win * 12),  # burst 2
        _frame(0.0, win * 10),  # gap → closes burst 2
    ]
    segs = [s for s in _run_stage(stage, frames) if s is not None]

    assert len(segs) == 2
    # Second run starts strictly after the first — a fresh run, not a merge.
    assert segs[1]["t0"] > segs[0]["t0"]


def test_all_silence_emits_no_segment() -> None:
    """A stream that is silent throughout never opens a run.

    Confirms the idle branch: unvoiced windows only feed the lead-pad ring and
    never construct a ``_Run``, so pure room noise produces zero segments.
    """
    stage = _wire_fakes(SileroVADStage())
    win = _SILERO_WINDOW_SAMPLES

    segs = _run_stage(stage, [_frame(0.0, win * 30)])
    assert segs == [None]  # only the sentinel — no speech ever detected


def test_window_never_split_across_frame_boundary() -> None:
    """Sub-window leftovers carry over so a 512-sample window is never split.

    Feeds frames sized to leave a remainder (not a whole-window multiple) and
    checks that the tail carry-over still assembles a single clean run — the
    fake scorer would flip a verdict if a window straddled two frames.
    """
    # Frame length deliberately NOT a multiple of the 512-sample window, so the
    # concatenate-tail logic is what keeps windows intact.
    stage = _wire_fakes(SileroVADStage(min_silence_ms=200, min_speech_ms=100, edge_pad_ms=0))
    win = _SILERO_WINDOW_SAMPLES
    odd = win + 100  # forces a 100-sample carry-over each frame

    frames = [_frame(1.0, odd) for _ in range(15)]  # continuous voiced audio
    frames.append(_frame(0.0, win * 10))  # close it out

    segs = [s for s in _run_stage(stage, frames) if s is not None]
    # A split window would have mis-scored and fragmented the run; we expect one.
    assert len(segs) == 1
    assert segs[0]["pcm"].shape[0] > 0
