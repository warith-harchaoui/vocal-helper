"""
vocal_helper.sources
====================

Async iterators that produce :class:`vocal_helper.types.PcmFrame`
events from a few canonical inputs.

Three sources ship in v0.1.0 :

- :func:`from_microphone` — wraps ``capture_helper.iter_mic_audio``
  (optional dependency ; requires ``vocal-helper[mic]``).
- :func:`from_wav_file` — replays a 16 kHz mono WAV at
  ``real_time=True`` (default) so the downstream stages see the
  same pacing as a live stream, or as fast as possible when
  ``real_time=False`` (for offline batch tests).
- :func:`from_numpy_array` — yields frames from an existing PCM
  buffer ; useful for unit tests and synthetic streams.

The contract is the same in all three cases :

    async for frame in source(...):
        # frame["pcm"].shape == (frame_samples,)
        # frame["sample_rate"] == 16000
        # frame["t0"] == seconds since first frame
        await queue.put(frame)

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vocal_helper.types import PcmFrame

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_FRAME_MS = 20


async def from_microphone(
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_ms: int = DEFAULT_FRAME_MS,
    device_name: str | None = None,
    device_index: int | None = None,
) -> AsyncIterator[PcmFrame]:
    """Yield PCM frames from the system's default (or named) microphone.

    Requires the ``mic`` extra (``pip install vocal-helper[mic]``).

    Parameters
    ----------
    sample_rate : int
        Target rate in Hz. Default 16 kHz.
    frame_ms : int
        Frame length in milliseconds. 20 ms is the default — matches
        capture_helper and lines up with Silero VAD's 32 ms window
        cleanly.
    device_name : str, optional
        Substring of the device name to pick.
    device_index : int, optional
        Concrete device index (taken from ``capture_helper.list_sources``).
    """
    try:
        import capture_helper as ch  # type: ignore
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "from_microphone requires capture-helper. "
            "Install with `pip install vocal-helper[mic]`."
        ) from e

    source = ch.pick_source(
        "microphone",
        name_substring=device_name,
        index=device_index,
    )
    start: float | None = None
    async for f in ch.iter_mic_audio(
        source, target_sample_rate=sample_rate, frame_ms=frame_ms, to_mono=True
    ):
        now = time.monotonic()
        if start is None:
            start = now
        yield PcmFrame(
            t0=now - start,
            sample_rate=sample_rate,
            pcm=np.asarray(f["pcm"], dtype=np.float32),
        )


async def from_wav_file(
    path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_ms: int = DEFAULT_FRAME_MS,
    real_time: bool = True,
) -> AsyncIterator[PcmFrame]:
    """Yield PCM frames from a mono WAV file.

    Pacing :

    - ``real_time=True`` — sleep ``frame_ms`` between frames so the
      downstream pipeline sees the same cadence as a live source.
    - ``real_time=False`` — yield as fast as possible. Useful for
      benchmarking and unit tests.
    """
    # Local import — soundfile is a dev dependency, not a runtime one
    # so the from_microphone / from_numpy_array paths stay light.
    import soundfile as sf  # type: ignore

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != sample_rate:
        raise ValueError(
            f"from_wav_file expects sample_rate={sample_rate} Hz, got {sr}. "
            f"Resample upstream with audio_helper.sound_converter."
        )

    frame_samples = sample_rate * frame_ms // 1000
    n = audio.shape[0]
    frame_period_s = frame_ms / 1000.0
    start = time.monotonic()
    cursor = 0
    while cursor < n:
        block = audio[cursor : cursor + frame_samples]
        if block.shape[0] < frame_samples:
            # Right-pad the tail frame with zeros to keep the contract.
            pad = np.zeros(frame_samples - block.shape[0], dtype=np.float32)
            block = np.concatenate([block, pad], axis=0)
        t0 = cursor / float(sample_rate)
        yield PcmFrame(
            t0=t0,
            sample_rate=sample_rate,
            pcm=block,
        )
        cursor += frame_samples
        if real_time:
            target = start + (cursor / float(sample_rate))
            sleep_s = target - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)


async def from_numpy_array(
    pcm: NDArray[np.float32],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_ms: int = DEFAULT_FRAME_MS,
    real_time: bool = False,
) -> AsyncIterator[PcmFrame]:
    """Yield PCM frames from an in-memory mono float32 buffer."""
    if pcm.ndim != 1:
        raise ValueError(f"from_numpy_array expects mono PCM, got shape {pcm.shape}")
    if pcm.dtype != np.float32:
        pcm = pcm.astype(np.float32, copy=False)
    frame_samples = sample_rate * frame_ms // 1000
    n = pcm.shape[0]
    start = time.monotonic()
    cursor = 0
    while cursor < n:
        block = pcm[cursor : cursor + frame_samples]
        if block.shape[0] < frame_samples:
            pad = np.zeros(frame_samples - block.shape[0], dtype=np.float32)
            block = np.concatenate([block, pad], axis=0)
        yield PcmFrame(
            t0=cursor / float(sample_rate),
            sample_rate=sample_rate,
            pcm=block,
        )
        cursor += frame_samples
        if real_time:
            target = start + (cursor / float(sample_rate))
            sleep_s = target - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
