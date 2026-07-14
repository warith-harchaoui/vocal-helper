"""
vocal_helper.sources
====================

Async iterators that produce :class:`vocal_helper.types.PcmFrame`
events from a few canonical inputs.

Four sources ship in v0.1.0 :

- :func:`from_microphone` — wraps ``capture_helper.iter_mic_audio``
  (optional dependency ; requires ``vocal-helper[mic]``).
- :func:`from_url` — wraps ``podcast_helper.extract_audio_stream``
  to consume any URL ``yt-dlp`` can reach (YouTube / Vimeo / Twitch
  VOD or live, SoundCloud, podcast RSS feeds, direct HLS / m3u8,
  direct audio files) as a paced PCM stream
  (optional dependency ; requires ``vocal-helper[stream]``).
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
            "from_microphone requires capture-helper. Install with `pip install vocal-helper[mic]`."
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


async def from_url(
    url: str,
    *,
    realtime: bool = True,
    speed: float = 1.0,
    headers: dict[str, str] | None = None,
    cookies_from_browser: str | None = None,
    record_to: str | Path | None = None,
) -> AsyncIterator[PcmFrame]:
    """Yield PCM frames from any URL (YouTube / Twitch / RSS / direct audio).

    Async wrapper over :func:`podcast_helper.extract_audio_stream` that
    **enforces** the speech-pipeline contract every downstream stage
    relies on. We deliberately do NOT expose ``sample_rate``,
    ``frame_ms`` or ``to_mono`` kwargs : vocal-helper hardcodes the
    only configuration that keeps Silero VAD, pyannote/embedding and
    whisper.cpp all happy at once.

    The enforced contract
    ---------------------

    * ``target_sample_rate = 16_000`` — Silero VAD ONNX v5 and
      pyannote/embedding are both trained at 16 kHz. Any other value
      would force re-sampling inside the stages.
    * ``to_mono = True`` — pyannote, whisper and the VAD all expect a
      single channel.
    * ``frame_ms = 20`` — 320 samples ; aligns with Silero's native
      32 ms window with one carry-over slice, so no re-buffering.
    * ``dtype = float32`` ∈ [-1, +1] — what whisper.cpp and pyannote
      ingest natively. ffmpeg's ``libswresample`` applies the
      Shannon-Nyquist anti-aliasing low-pass at the new Nyquist when
      down-sampling, so spectral aliasing is impossible.

    Callers that need a different shape should drop the helper and
    call ``podcast_helper.extract_audio_stream`` directly — those
    callers are by definition outside the speech pipeline.

    Requires ``podcast-helper`` (``pip install vocal-helper[stream]``)
    plus ``ffmpeg`` and ``yt-dlp`` on PATH.

    Parameters
    ----------
    url : str
        File path, ``file://`` URL, direct audio URL (MP3 / M4A / Opus
        / WAV / HLS m3u8), RSS feed URL (auto-picks latest episode),
        or any ``yt-dlp``-supported URL — YouTube, Vimeo, SoundCloud,
        Twitch VOD and live, etc. Spotify-protected and Apple Podcasts
        catalog URLs raise ``NotImplementedError`` with hints.
    realtime : bool, default True
        Pace decoding at wall-clock (ffmpeg's ``-re``). Set ``False``
        for burst-decoding a VOD (offline benchmarking). Live sources
        pace themselves, so podcast-helper forces this to ``False``
        internally.
    speed : float, default 1.0
        Playback rate for **VOD only**. Implemented via ffmpeg's
        ``atempo=`` filter so pitch is preserved. ``2.0`` doubles ASR
        throughput on long episodes ; ``0.5`` slows down for proofing.
        Raises ``ValueError`` on live streams.
    headers : dict[str, str], optional
        HTTP headers ffmpeg should send. Merged on top of yt-dlp's
        per-source headers (your keys win).
    cookies_from_browser : str, optional
        ``"firefox"`` / ``"chrome"`` / ``"safari"`` / etc. — used by
        yt-dlp for age-gated, members-only or private content.
    record_to : str or Path, optional
        If set, ffmpeg writes a parallel compressed archive of the
        same audio to this path while the live PCM stream is consumed.
        Single decode, two encoder paths. See ``podcast_helper`` docs
        for the codec-by-extension table.

    Yields
    ------
    PcmFrame
        ``t0`` in seconds since the source started,
        ``sample_rate == 16_000``, ``pcm`` mono ``float32`` of length
        320 (= 16 000 × 20 / 1 000).

    Raises
    ------
    RuntimeError
        If the upstream frame violates the speech contract — wrong
        sample rate, wrong dtype, non-mono. Fail-loud is the point :
        a silent contract drift would corrupt every downstream stage.

    Examples
    --------
    >>> import asyncio, vocal_helper as voh
    >>> async def main():
    ...     pipeline = voh.Pipeline(
    ...         source=lambda: voh.sources.from_url(
    ...             "https://www.youtube.com/watch?v=YE7VzlLtp-4",
    ...         ),
    ...         config=voh.PipelineConfig(
    ...             diar={"backend": "pyannote"},
    ...             llm={"model": "gemma4:e4b"},
    ...         ),
    ...     )
    ...     async for ev in pipeline.run():
    ...         if isinstance(ev, dict) and "text" in ev:
    ...             print(f"[{ev['t0']:.1f}s {ev['speaker']}] {ev['text']}")
    >>> asyncio.run(main())  # doctest: +SKIP
    """
    try:
        import podcast_helper as ph  # type: ignore
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "from_url requires podcast-helper. Install with `pip install vocal-helper[stream]`."
        ) from e

    expected_frame_samples = DEFAULT_SAMPLE_RATE * DEFAULT_FRAME_MS // 1000

    # podcast-helper's frame schema: {t_abs_s, pcm, voiced}. vocal-helper's
    # PcmFrame uses {t0, sample_rate, pcm}. Repack at the boundary so
    # downstream stages keep their existing typed contract.
    async for f in ph.extract_audio_stream(
        url,
        target_sample_rate=DEFAULT_SAMPLE_RATE,
        to_mono=True,
        realtime=realtime,
        frame_ms=DEFAULT_FRAME_MS,
        headers=headers,
        cookies_from_browser=cookies_from_browser,
        speed=speed,
        record_to=str(record_to) if record_to is not None else None,
    ):
        pcm = np.asarray(f["pcm"])
        # Validate the contract loudly. We pinned the request
        # (16 kHz / mono / 20 ms / float32) — if the producer returns
        # anything else, fail here rather than corrupt downstream VAD /
        # diar / ASR with bad-shape input.
        if pcm.dtype != np.float32:
            raise RuntimeError(
                f"from_url contract violated : expected float32 PCM, got "
                f"{pcm.dtype}. podcast_helper.extract_audio_stream is "
                "documented to emit float32 ; check the installed version."
            )
        if pcm.ndim != 1:
            raise RuntimeError(
                f"from_url contract violated : expected mono PCM, got "
                f"shape {pcm.shape}. We called extract_audio_stream with "
                "to_mono=True ; this should not happen."
            )
        # Tail frame may be short ; full frames must be ≤ 320 samples.
        if pcm.shape[0] > expected_frame_samples:
            raise RuntimeError(
                f"from_url contract violated : expected ≤ "
                f"{expected_frame_samples} samples per frame (we asked "
                f"for {DEFAULT_FRAME_MS} ms at {DEFAULT_SAMPLE_RATE} Hz), "
                f"got {pcm.shape[0]}."
            )
        yield PcmFrame(
            t0=float(f["t_abs_s"]),
            sample_rate=DEFAULT_SAMPLE_RATE,
            pcm=pcm,
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
    # Decode via audio-helper (ffmpeg-backed): ANY format/codec — mp3, m4a,
    # opus, flac, or a video's audio track — auto-resampled to sample_rate and
    # down-mixed to mono. No soundfile, no "must be a 16 kHz WAV" precondition.
    from audio_helper import load_audio

    audio, sr = load_audio(str(path), target_sample_rate=sample_rate, to_mono=True, to_numpy=True)
    audio = np.asarray(audio, dtype=np.float32)

    frame_samples = sample_rate * frame_ms // 1000
    n = audio.shape[0]
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
