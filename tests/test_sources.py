"""Unit tests for ``vocal_helper.sources`` — offline, no mic / network.

Every async test uses :func:`asyncio.run` so the loop policy stays
explicit ; we run with ``real_time=False`` everywhere so the suite
completes in a few milliseconds.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

import vocal_helper as vh

# ---------------------------------------------------------------------------
# from_numpy_array
# ---------------------------------------------------------------------------


def test_from_numpy_array_yields_correct_frame_count() -> None:
    """50 frames of 20 ms cover 1 s @ 16 kHz exactly."""
    pcm = np.zeros(16_000, dtype=np.float32)

    async def collect() -> list[vh.PcmFrame]:
        return [f async for f in vh.sources.from_numpy_array(pcm)]

    frames = asyncio.run(collect())
    assert len(frames) == 50
    assert all(f["pcm"].shape == (320,) for f in frames)
    assert all(f["sample_rate"] == 16_000 for f in frames)


def test_from_numpy_array_pads_last_frame() -> None:
    """A buffer that isn't a clean multiple of the frame size is zero-padded."""
    # 17 ms → not a multiple of 20 ms ; expect 1 frame zero-padded.
    pcm = np.ones(272, dtype=np.float32)  # 17 ms @ 16 kHz

    async def collect() -> list[vh.PcmFrame]:
        return [f async for f in vh.sources.from_numpy_array(pcm)]

    frames = asyncio.run(collect())
    assert len(frames) == 1
    block = frames[0]["pcm"]
    assert block.shape == (320,)
    assert np.all(block[:272] == 1.0)
    assert np.all(block[272:] == 0.0)


def test_from_numpy_array_rejects_stereo() -> None:
    """``from_numpy_array`` is mono-only ; shape (n, 2) must raise."""
    stereo = np.zeros((16_000, 2), dtype=np.float32)

    async def consume() -> None:
        async for _ in vh.sources.from_numpy_array(stereo):
            pass

    with pytest.raises(ValueError, match="mono"):
        asyncio.run(consume())


def test_from_numpy_array_casts_dtype() -> None:
    """A float64 buffer is silently downcast to float32, not rejected."""
    pcm = np.zeros(800, dtype=np.float64)  # 50 ms @ 16 kHz

    async def collect() -> list[vh.PcmFrame]:
        return [f async for f in vh.sources.from_numpy_array(pcm)]

    frames = asyncio.run(collect())
    assert frames[0]["pcm"].dtype == np.float32


def test_from_numpy_array_t0_is_monotonic() -> None:
    """Frame timestamps must be sorted strictly increasing."""
    pcm = np.zeros(16_000, dtype=np.float32)

    async def collect() -> list[float]:
        return [f["t0"] async for f in vh.sources.from_numpy_array(pcm)]

    times = asyncio.run(collect())
    assert times == sorted(times)
    assert len(set(times)) == len(times)


# ---------------------------------------------------------------------------
# from_wav_file
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_wav(tmp_path: Path) -> Path:
    """Generate a 100 ms mono float32 16 kHz WAV on disk."""
    sf = pytest.importorskip("soundfile")
    pcm = np.linspace(-1, 1, 1_600, dtype=np.float32)  # 100 ms
    path = tmp_path / "tiny.wav"
    sf.write(str(path), pcm, 16_000, subtype="PCM_16")
    return path


def test_from_wav_file_round_trip(tiny_wav: Path) -> None:
    """A 100 ms WAV at 16 kHz / 20 ms frames → 5 frames."""
    async def collect() -> list[vh.PcmFrame]:
        return [f async for f in vh.sources.from_wav_file(tiny_wav, real_time=False)]

    frames = asyncio.run(collect())
    assert len(frames) == 5
    assert all(f["pcm"].shape == (320,) for f in frames)
    # First sample should be near -1 (start of the linspace).
    assert frames[0]["pcm"][0] < -0.95


def test_from_wav_file_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    """8 kHz WAV must trip the explicit guard, not silently resample."""
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "wrong_sr.wav"
    sf.write(str(path), np.zeros(800, dtype=np.float32), 8_000, subtype="PCM_16")

    async def consume() -> None:
        async for _ in vh.sources.from_wav_file(path, real_time=False):
            pass

    with pytest.raises(ValueError, match="sample_rate"):
        asyncio.run(consume())


def test_from_wav_file_handles_multichannel(tmp_path: Path) -> None:
    """Stereo input is averaged to mono ; the contract still holds."""
    sf = pytest.importorskip("soundfile")
    stereo = np.zeros((1_600, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, 16_000, subtype="PCM_16")

    async def collect() -> list[vh.PcmFrame]:
        return [f async for f in vh.sources.from_wav_file(path, real_time=False)]

    frames = asyncio.run(collect())
    # Stereo averages to ~0 ; PCM_16 quantisation can leak ~ ±1 LSB.
    assert all(np.abs(f["pcm"]).max() <= 1e-3 for f in frames)


# ---------------------------------------------------------------------------
# Optional-dep gating
# ---------------------------------------------------------------------------


def test_from_microphone_raises_clear_error_without_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_microphone`` must surface an actionable ImportError if the
    ``mic`` extra isn't installed, not crash deeper in the stack."""
    import sys

    # If capture_helper is already importable on this box, simulate the
    # missing extra by hiding it.
    monkeypatch.setitem(sys.modules, "capture_helper", None)

    async def consume() -> None:
        async for _ in vh.sources.from_microphone():
            pass

    with pytest.raises(ImportError, match=r"vocal-helper\[mic\]"):
        asyncio.run(consume())
