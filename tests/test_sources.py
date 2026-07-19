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
from pathlib import Path

import numpy as np
import pytest

import vocal_helper as voh


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
