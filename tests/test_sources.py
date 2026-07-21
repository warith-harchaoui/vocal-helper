"""Unit tests for ``vocal_helper.sources`` — offline, no mic / network.

Every async test uses :func:`asyncio.run` so the loop policy stays
explicit ; we run with ``real_time=False`` everywhere so the suite
completes in a few milliseconds. These tests are model-free — they
push numpy buffers and tiny on-disk WAVs through the framing code and
assert the emitted :class:`~vocal_helper.PcmFrame` sequence, plus the
``probe_duration_s`` metadata read the router depends on.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import vocal_helper as voh
from vocal_helper import sources


def _collect(source) -> list[voh.PcmFrame]:
    """Drain an async PCM source to a list on a fresh event loop.

    Parameters
    ----------
    source : AsyncIterator[vocal_helper.PcmFrame]
        Any source coroutine-generator (numpy / WAV) to exhaust.

    Returns
    -------
    list of vocal_helper.PcmFrame
        Every frame the source yielded, in order.
    """

    async def _drain() -> list[voh.PcmFrame]:
        """Gather every frame the source yields."""
        return [f async for f in source]

    return asyncio.run(_drain())


# ---------------------------------------------------------------------------
# from_numpy_array — framing contract
# ---------------------------------------------------------------------------


def test_from_numpy_array_framing_and_timestamps() -> None:
    """A whole-second mono buffer frames cleanly with a monotonic clock.

    Exercises the happy path of :func:`from_numpy_array` in one flow:
    frame count (50 × 20 ms == 1 s), per-frame shape / sample-rate, and
    strictly increasing ``t0`` timestamps.
    """
    # 1 s @ 16 kHz → exactly 50 frames of 320 samples (20 ms), no padding.
    pcm = np.zeros(16_000, dtype=np.float32)
    frames = _collect(voh.sources.from_numpy_array(pcm))

    assert len(frames) == 50
    assert all(f["pcm"].shape == (320,) for f in frames)
    assert all(f["sample_rate"] == 16_000 for f in frames)

    # Timestamps are the stream clock: sorted and strictly increasing.
    times = [f["t0"] for f in frames]
    assert times == sorted(times)
    assert len(set(times)) == len(times)


def test_from_numpy_array_pads_casts_and_rejects_stereo() -> None:
    """Sub-frame float64 buffers pad + downcast ; stereo is rejected.

    Folds three edge-path guarantees of :func:`from_numpy_array`: a
    buffer that is not a clean multiple of the frame size is zero-padded
    to fill the trailing frame, a non-float32 dtype is silently downcast
    rather than rejected, and a (n, 2) stereo buffer raises — the source
    is mono-only.
    """
    # 17 ms @ 16 kHz (272 samples) as float64 → 1 frame, padded + downcast.
    pcm = np.ones(272, dtype=np.float64)
    frames = _collect(voh.sources.from_numpy_array(pcm))

    assert len(frames) == 1
    block = frames[0]["pcm"]
    # Padding: real samples preserved, tail zero-filled to 320.
    assert block.shape == (320,)
    assert np.all(block[:272] == 1.0)
    assert np.all(block[272:] == 0.0)
    # Cast: float64 in, float32 out.
    assert block.dtype == np.float32

    # Mono-only guard: a stereo buffer must raise, not silently downmix.
    stereo = np.zeros((16_000, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="mono"):
        _collect(voh.sources.from_numpy_array(stereo))


# ---------------------------------------------------------------------------
# from_wav_file — decode + normalise on-disk audio
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_wav(tmp_path: Path) -> Path:
    """Generate a 100 ms mono float32 16 kHz WAV on disk.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.

    Returns
    -------
    pathlib.Path
        Path to a 100 ms ``linspace(-1, 1)`` mono WAV.
    """
    import scipy.io.wavfile as wavfile

    pcm = np.linspace(-1, 1, 1_600, dtype=np.float32)  # 100 ms
    path = tmp_path / "tiny.wav"
    wavfile.write(str(path), 16_000, pcm)
    return path


def test_from_wav_file_round_trip_and_probe(tiny_wav: Path) -> None:
    """A 100 ms 16 kHz mono WAV decodes to 5 frames and probes to ≈ 0.1 s.

    Covers the nominal on-disk decode path (frame count, per-frame shape,
    first sample tracking the start of the ``linspace`` ramp ≈ -1) plus
    the ``probe_duration_s`` metadata read the router relies on: it must
    return the real length for a readable file and fall back to ``None``
    — never raise — for a missing path.
    """
    frames = _collect(voh.sources.from_wav_file(tiny_wav, real_time=False))

    assert len(frames) == 5
    assert all(f["pcm"].shape == (320,) for f in frames)
    # First sample sits at the start of the -1 → 1 ramp.
    assert frames[0]["pcm"][0] < -0.95

    # Router pre-flight: cheap metadata read yields the true 100 ms length,
    # and any unreadable path degrades to None ("unknown length").
    seconds = voh.sources.probe_duration_s(tiny_wav)
    assert seconds is not None
    assert seconds == pytest.approx(0.1, abs=0.01)
    assert voh.sources.probe_duration_s(tiny_wav.with_name("missing.wav")) is None


def test_from_wav_file_normalises_rate_and_channels(tmp_path: Path) -> None:
    """Wrong sample rate is resampled and stereo is averaged to mono.

    Both files carry 100 ms of audio: an 8 kHz mono buffer (transparently
    resampled to 16 kHz via audio-helper) and a 16 kHz stereo buffer with
    opposite-sign channels (averaged to ≈ 0). Either way the source must
    emit 5 frames of 320 samples that satisfy the 16 kHz mono contract.
    """
    import scipy.io.wavfile as wavfile

    # 8 kHz mono → resampled up to 16 kHz.
    sr8k = tmp_path / "sr8k.wav"
    wavfile.write(str(sr8k), 8_000, np.zeros(800, dtype=np.float32))

    # 16 kHz stereo with ±0.5 channels → averaged to ~0 (±1 LSB PCM_16 leak).
    stereo = np.zeros((1_600, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    stereo_path = tmp_path / "stereo.wav"
    wavfile.write(str(stereo_path), 16_000, stereo)

    resampled = _collect(voh.sources.from_wav_file(sr8k, real_time=False))
    assert len(resampled) == 5
    assert all(f["pcm"].shape == (320,) for f in resampled)

    averaged = _collect(voh.sources.from_wav_file(stereo_path, real_time=False))
    assert len(averaged) == 5
    assert all(np.abs(f["pcm"]).max() <= 1e-3 for f in averaged)


# ---------------------------------------------------------------------------
# Optional-dep gating
# ---------------------------------------------------------------------------


def test_from_microphone_raises_clear_error_without_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_microphone`` surfaces an actionable ImportError without the extra.

    When the ``mic`` extra isn't installed it must point at
    ``vocal-helper[mic]`` rather than crashing deeper in the stack.
    """
    import sys

    # If capture_helper is importable on this box, hide it to simulate the
    # missing extra.
    monkeypatch.setitem(sys.modules, "capture_helper", None)

    async def consume() -> None:
        """Iterate the mic source so the missing-extra ImportError fires."""
        async for _ in voh.sources.from_microphone():
            pass

    with pytest.raises(ImportError, match=r"vocal-helper\[mic\]"):
        asyncio.run(consume())


def test_from_url_raises_clear_error_without_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_url`` surfaces an actionable ImportError without the stream extra.

    Symmetric to the mic guard: with ``podcast_helper`` absent the source must
    point at ``vocal-helper[stream]`` instead of crashing on a bare import deep
    in the async generator.
    """
    # Hide podcast_helper (installed or not) so the missing-extra path fires.
    monkeypatch.setitem(sys.modules, "podcast_helper", None)

    async def consume() -> None:
        """Iterate the URL source so the missing-extra ImportError fires."""
        async for _ in voh.sources.from_url("https://example.com/a.mp3"):
            pass

    with pytest.raises(ImportError, match=r"vocal-helper\[stream\]"):
        asyncio.run(consume())


# ---------------------------------------------------------------------------
# from_url — the fail-loud contract on a fake podcast_helper stream
# ---------------------------------------------------------------------------


def _install_fake_podcast_helper(monkeypatch: pytest.MonkeyPatch, frames: list[dict]) -> None:
    """Inject a stub ``podcast_helper`` whose stream yields ``frames`` verbatim.

    ``from_url`` is documented to *enforce* the 16 kHz / mono / float32 / <=320
    contract on whatever the producer emits. Stubbing the producer lets us feed
    deliberately-malformed frames and prove the guard rails fire, without a real
    yt-dlp / ffmpeg decode.
    """

    async def _extract_audio_stream(url: str, **kwargs: object):
        """Replay the scripted frames as podcast_helper's {t_abs_s, pcm, voiced}."""
        for f in frames:
            yield f

    fake = types.ModuleType("podcast_helper")
    fake.extract_audio_stream = _extract_audio_stream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "podcast_helper", fake)


@pytest.mark.parametrize(
    ("bad_pcm", "needle"),
    [
        # Wrong dtype: producer must emit float32, not int16.
        (np.zeros(320, dtype=np.int16), "float32"),
        # Non-mono: a 2-D (stereo) frame violates the mono contract.
        (np.zeros((320, 2), dtype=np.float32), "mono"),
        # Oversized: a full frame must be <= 320 samples at 16 kHz / 20 ms.
        (np.zeros(321, dtype=np.float32), "320"),
    ],
)
def test_from_url_fails_loud_on_contract_violation(
    monkeypatch: pytest.MonkeyPatch,
    bad_pcm: np.ndarray,
    needle: str,
) -> None:
    """A producer frame that breaks the speech contract raises RuntimeError, not silence.

    Sweeps the three guard rails ``from_url`` enforces on every inbound frame —
    dtype (float32), rank (mono), and size (<= 320 samples). Each violation must
    fail loud at the boundary so a shape drift can never corrupt the downstream
    VAD / diar / ASR stages; the message names the offending property.

    Parameters
    ----------
    bad_pcm : numpy.ndarray
        A malformed PCM buffer that trips exactly one guard.
    needle : str
        Substring the raised RuntimeError message must contain.
    """
    _install_fake_podcast_helper(monkeypatch, [{"t_abs_s": 0.0, "pcm": bad_pcm, "voiced": True}])

    async def consume() -> None:
        """Pull the first (malformed) frame so the contract check fires."""
        async for _ in voh.sources.from_url("https://example.com/a.mp3"):
            pass

    with pytest.raises(RuntimeError, match=needle):
        asyncio.run(consume())


def test_from_url_repacks_valid_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contract-conforming frames are repacked into PcmFrames with t0 from t_abs_s.

    The happy path of the boundary: a full 320-sample float32 frame and a short
    tail frame (both legal) pass the guards and are re-shaped from the producer's
    ``{t_abs_s, pcm, voiced}`` into vocal-helper's ``{t0, sample_rate, pcm}``,
    carrying the absolute timestamp through as ``t0``.
    """
    frames = [
        {"t_abs_s": 0.0, "pcm": np.ones(320, dtype=np.float32), "voiced": True},
        {"t_abs_s": 0.02, "pcm": np.ones(100, dtype=np.float32), "voiced": True},  # short tail
    ]
    _install_fake_podcast_helper(monkeypatch, frames)

    out = _collect(voh.sources.from_url("https://example.com/a.mp3"))

    assert len(out) == 2
    assert all(f["sample_rate"] == 16_000 for f in out)
    # t0 is taken straight from the producer's absolute timestamp.
    assert out[0]["t0"] == pytest.approx(0.0)
    assert out[1]["t0"] == pytest.approx(0.02)
    # The short tail frame is passed through as-is (not padded here).
    assert out[1]["pcm"].shape == (100,)


# ---------------------------------------------------------------------------
# probe_duration_s — router pre-flight degradation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reading", [0.0, -1.0])
def test_probe_duration_non_positive_reads_as_unknown(
    monkeypatch: pytest.MonkeyPatch, reading: float
) -> None:
    """A zero / negative duration read degrades to None, never a bogus length.

    The router treats ``None`` as "unknown length" and routes to the robust
    long-form branch. A container that reports 0 s (or a negative sentinel) is as
    good as unknown, so ``probe_duration_s`` must not forward it — a false short
    length would mis-route to the duration-capped backend.

    Parameters
    ----------
    reading : float
        The non-positive duration the probe backend returns.
    """
    # Patch the audio_helper probe at its import site inside sources.py.
    import audio_helper

    monkeypatch.setattr(audio_helper, "get_audio_duration", lambda _p: reading)
    assert sources.probe_duration_s("whatever.wav") is None


def test_probe_duration_swallows_backend_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any probe backend exception degrades to None rather than propagating.

    ``probe_duration_s`` is a best-effort pre-flight; a raising backend (missing
    ffprobe, unreadable container) must be caught and reported as "unknown
    length" so the router can still make a decision.
    """
    import audio_helper

    def _boom(_p: str) -> float:
        """Simulate a probe backend blowing up mid-read."""
        raise RuntimeError("ffprobe exploded")

    monkeypatch.setattr(audio_helper, "get_audio_duration", _boom)
    assert sources.probe_duration_s("whatever.wav") is None


# ---------------------------------------------------------------------------
# empty-input edge cases — the chunkers must not emit spurious frames
# ---------------------------------------------------------------------------


def test_empty_numpy_buffer_yields_no_frames() -> None:
    """A zero-length mono buffer produces an empty frame stream, not a padded frame.

    The cursor chunker's ``while cursor < n`` loop must never enter for ``n == 0``
    — an empty input is a legitimate silent stream, and fabricating a padded
    frame would inject phantom audio into the pipeline.
    """
    empty = np.zeros(0, dtype=np.float32)
    assert _collect(voh.sources.from_numpy_array(empty)) == []


def test_empty_wav_file_yields_no_frames(tmp_path: Path) -> None:
    """A WAV with zero audio samples decodes to zero frames.

    Symmetric to the empty-array case but through the on-disk decode path:
    audio-helper returns an empty buffer and the chunker emits nothing rather
    than a single zero-padded frame.
    """
    import scipy.io.wavfile as wavfile

    empty_path = tmp_path / "empty.wav"
    wavfile.write(str(empty_path), 16_000, np.zeros(0, dtype=np.float32))

    assert _collect(voh.sources.from_wav_file(empty_path, real_time=False)) == []
