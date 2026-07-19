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


def probe_duration_s(path: str | Path) -> float | None:
    """Best-effort media duration in seconds, for the diarization router.

    The router (:func:`vocal_helper.router.select_diarization`) needs the audio
    length to choose the offline backend — short/dense → NeMo Sortformer, long →
    pyannote. This is a *cheap metadata read* (ffprobe via ``audio_helper``), not
    a full decode, so it is safe to call before building the pipeline. Any
    failure (unreadable file, missing probe backend) returns ``None``, which the
    router treats as "unknown length" and routes to the robust long-form branch.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to any ffmpeg-decodable media file (wav / mp3 / m4a / video / …).

    Returns
    -------
    float or None
        Duration in seconds when it can be read and is positive, else ``None``.

    Examples
    --------
    >>> probe_duration_s("/nonexistent.wav") is None
    True
    """
    # ffprobe (via audio_helper) reads container metadata without decoding the
    # whole stream — O(1) regardless of file length, and cross-format.
    try:
        from audio_helper import get_audio_duration

        seconds = float(get_audio_duration(str(path)))
    except Exception:  # noqa: BLE001 — any probe failure ⇒ "unknown length"
        return None
    # A zero/negative reading is as good as unknown; don't feed it to the router.
    return seconds if seconds > 0 else None


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
    # Lazy import : the mic backend is an optional extra, so keep it out of
    # the import path for callers that only stream from files or URLs.
    try:
        import capture_helper as ch  # type: ignore
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "from_microphone requires capture-helper. Install with `pip install vocal-helper[mic]`."
        ) from e

    # Resolve the concrete device : name substring / index narrow it down,
    # otherwise capture-helper falls back to the system default input.
    source = ch.pick_source(
        "microphone",
        name_substring=device_name,
        index=device_index,
    )
    # Rebase timestamps to the first frame so ``t0`` is "seconds since
    # capture started", independent of the machine's monotonic epoch.
    start: float | None = None
    async for f in ch.iter_mic_audio(
        source, target_sample_rate=sample_rate, frame_ms=frame_ms, to_mono=True
    ):
        now = time.monotonic()
        # First frame establishes the zero point for every later t0.
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
    # Lazy import : the streaming backend (yt-dlp + ffmpeg) is an optional
    # extra ; only pull it in when a caller actually streams from a URL.
    try:
        import podcast_helper as ph  # type: ignore
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "from_url requires podcast-helper. Install with `pip install vocal-helper[stream]`."
        ) from e

    # The upper bound a well-behaved producer must respect : 320 samples at
    # 16 kHz / 20 ms. Precompute once to validate every frame cheaply below.
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
    # Force float32 : the whole pipeline (VAD / diar / whisper) ingests
    # float32 PCM, so normalise here rather than let dtype drift downstream.
    audio = np.asarray(audio, dtype=np.float32)

    # Walk the buffer in fixed frame_samples-sized blocks (a "cursor" chunker).
    frame_samples = sample_rate * frame_ms // 1000
    n = audio.shape[0]
    # ``start`` anchors the real-time pacing clock ; ``cursor`` is the read head.
    start = time.monotonic()
    cursor = 0
    while cursor < n:
        block = audio[cursor : cursor + frame_samples]
        if block.shape[0] < frame_samples:
            # Right-pad the tail frame with zeros to keep the contract.
            pad = np.zeros(frame_samples - block.shape[0], dtype=np.float32)
            block = np.concatenate([block, pad], axis=0)
        # ``t0`` is derived from the sample offset, NOT wall-clock — exact and
        # reproducible whether we replay in real time or burst as fast as we can.
        t0 = cursor / float(sample_rate)
        yield PcmFrame(
            t0=t0,
            sample_rate=sample_rate,
            pcm=block,
        )
        cursor += frame_samples
        # Real-time replay : sleep until this frame's natural playout instant so
        # downstream stages see the same cadence a live source would produce.
        if real_time:
            target = start + (cursor / float(sample_rate))
            sleep_s = target - time.monotonic()
            # Only sleep if we're ahead ; if decoding lagged, don't sleep negative.
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
    # Fail loud on shape : a stereo/2-D array would silently mis-frame and
    # corrupt every downstream stage, so reject it up front.
    if pcm.ndim != 1:
        raise ValueError(f"from_numpy_array expects mono PCM, got shape {pcm.shape}")
    # Coerce dtype without copying when already float32 — cheap contract fix.
    if pcm.dtype != np.float32:
        pcm = pcm.astype(np.float32, copy=False)
    # Same cursor-based chunker as from_wav_file (kept parallel on purpose).
    frame_samples = sample_rate * frame_ms // 1000
    n = pcm.shape[0]
    start = time.monotonic()
    cursor = 0
    while cursor < n:
        block = pcm[cursor : cursor + frame_samples]
        if block.shape[0] < frame_samples:
            # Zero-pad the final short frame so consumers get uniform lengths.
            pad = np.zeros(frame_samples - block.shape[0], dtype=np.float32)
            block = np.concatenate([block, pad], axis=0)
        yield PcmFrame(
            # Sample-offset timestamp — deterministic regardless of pacing.
            t0=cursor / float(sample_rate),
            sample_rate=sample_rate,
            pcm=block,
        )
        cursor += frame_samples
        # ``real_time`` defaults False here (tests want burst speed) ; when set,
        # pace to playout so a synthetic stream mimics live cadence.
        if real_time:
            target = start + (cursor / float(sample_rate))
            sleep_s = target - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
