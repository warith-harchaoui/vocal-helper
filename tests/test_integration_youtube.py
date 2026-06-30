"""End-to-end integration tests against a real YouTube source.

Skipped by default — these tests pull yt-dlp, ffmpeg, whisper.cpp,
pyannote and a network connection. Run with::

    pytest -v -m integration tests/test_integration_youtube.py

Stack split (one responsibility per helper)
-------------------------------------------

* ``podcast-helper`` owns the **audio** path. ``vh.sources.from_url``
  is a thin async wrapper over
  ``podcast_helper.extract_audio_stream`` — URL → 16 kHz mono float32
  PCM frames. Same entry point for RSS, direct audio URLs, HLS and
  every yt-dlp source (including YouTube).
* ``youtube-helper`` owns the **image / metadata** path. We use it
  here for one thing : pulling the auto-generated captions as a
  WebVTT file so we have a reference transcript to compare against.
* ``vocal-helper`` owns the speech-processing pipeline (VAD → diar →
  ASR → optional LLM).

What we cover
-------------

1. ``test_youtube_captions_fetchable`` — sanity-check the reference
   side before the heavy ASR work : if the URL has no captions,
   later assertions can't be meaningful, fail fast.
2. ``test_streaming_pipeline_yields_utterance`` — short clip through
   the streaming :class:`vh.Pipeline` (Silero VAD + online diar +
   whisper.cpp). Asserts at least one non-empty utterance is emitted.
3. ``test_offline_pipeline_vs_youtube_captions`` — the headline
   test : run :class:`vh.OfflinePipeline` on a 60 s clip, compare
   the transcript with the YouTube auto-captions on the same window,
   require Jaccard overlap on word sets ≥ ``MIN_JACCARD``.
4. ``test_streaming_realtime_pacing`` — proves
   ``from_url(realtime=True)`` actually paces at wall-clock. We
   consume 5 s of audio and measure that it took 5 s (± slack), not
   the burst-decoded sub-second time.

Target URL : https://www.youtube.com/watch?v=FisrbY90td0 (~ 15 min).
We only consume the head slice so each test stays under a minute on
a warm model cache.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

import vocal_helper as vh
from vocal_helper._settings import resolve_hf_token

# ---------------------------------------------------------------------------
# Test parameters — knobs in one place.
# ---------------------------------------------------------------------------

VIDEO_URL = "https://www.youtube.com/watch?v=FisrbY90td0"
# How many seconds of audio to actually run through the pipeline.
# Kept short on purpose : pyannote 3.1 on CPU + whisper turbo on 30 s
# of speech stays under a minute on Apple Silicon, and short clips
# minimise the window for a ffmpeg / yt-dlp / HF-Hub stall to bite.
CLIP_S = 30.0
# Shorter window for the lightweight "does it start" check.
SMOKE_CLIP_S = 12.0
# Hard wall-clock budgets per phase. If we blow past them, something
# external (rate-limit retry, ffmpeg leak, stalled model download) is
# wrong — fail loudly with a clear message rather than block the suite.
INGEST_TIMEOUT_S = 90.0     # head download via podcast-helper
# 360 s covers a cold-cache scenario : pyannote/speaker-diarization-3.1
# loads segmentation-3.0 + embedding + clusterer (~ 60-90 s lazy on
# first call), whisper-turbo init Metal allocates ~ 800 MB and a
# warm-up forward pass takes another ~ 10 s, then the actual diar +
# ASR on CLIP_S of audio runs in 60-120 s on Apple M2 Max. Warm-cache
# runs come in well under a minute.
PIPELINE_TIMEOUT_S = 360.0
# Minimum lexical overlap between our transcript and YT's auto-captions
# (Jaccard on word sets, lowercased, punctuation stripped). Auto-captions
# and whisper.cpp are two noisy hypotheses of the same audio — they
# won't match perfectly. 0.20 is the floor below which something is
# clearly broken.
MIN_JACCARD = 0.20
# Sample rate the rest of the stack assumes everywhere.
SR = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _activate_hf_token() -> None:
    """Re-export ``$HF_TOKEN`` past the autouse conftest isolation."""
    import os

    from vocal_helper import _settings as s

    real_path = None
    for p in (
        Path.cwd() / "settings.yaml",
        Path(s.__file__).resolve().parent.parent / "settings.yaml",
    ):
        if p.is_file():
            real_path = p
            break
    token = os.environ.get("HF_TOKEN") or (
        real_path
        and s._parse_minimal_yaml(real_path.read_text()).get("secrets", {}).get(
            "hf_token"
        )
    )
    if token and token not in ("hf_XXXX", "hf_yourtoken"):
        os.environ["HF_TOKEN"] = token


def _parse_vtt_text(vtt_path: Path, max_seconds: float | None = None) -> str:
    """Return spoken text from a WebVTT file, optionally clipped."""
    text_parts: list[str] = []
    ts_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})\.\d{3}\s*-->\s*(\d{2}):(\d{2}):(\d{2})"
    )
    keep = True
    for raw_line in vtt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        m = ts_re.search(line)
        if m:
            cue_end = (
                int(m.group(4)) * 3600 + int(m.group(5)) * 60 + int(m.group(6))
            )
            if max_seconds is not None and cue_end > max_seconds:
                keep = False
            continue
        if not keep:
            continue
        # Strip YouTube colour / positional tags : ``<c.colorE0E0E0>`` etc.
        text_parts.append(re.sub(r"<[^>]+>", "", line))
    return " ".join(text_parts)


def _normalise_words(text: str) -> set[str]:
    """Lowercase, drop punctuation, keep accented letters and digits."""
    text = text.lower()
    words = re.findall(r"[a-zà-ÿ0-9]+(?:'[a-zà-ÿ0-9]+)?", text)
    return set(words)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


async def _clipped_url_source(
    url: str, max_s: float, *, realtime: bool = False,
) -> AsyncIterator[vh.PcmFrame]:
    """Wrap ``vh.sources.from_url`` and stop after ``max_s`` of audio."""
    accumulated_s = 0.0
    async for frame in vh.sources.from_url(url, realtime=realtime):
        yield frame
        accumulated_s += frame["pcm"].shape[0] / float(frame["sample_rate"])
        if accumulated_s >= max_s:
            return


# ---------------------------------------------------------------------------
# Module-scoped fixtures — heavy I/O happens once per session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def yh():
    """``youtube_helper`` for captions ; skip if absent."""
    return pytest.importorskip("youtube_helper")


@pytest.fixture(scope="module")
def ph():
    """``podcast_helper`` for the audio stream ; skip if absent."""
    return pytest.importorskip("podcast_helper")


@pytest.fixture(scope="module")
def hf_token() -> str:
    _activate_hf_token()
    token = resolve_hf_token()
    if not token:
        pytest.skip(
            "no HF_TOKEN / settings.yaml secrets.hf_token configured — "
            "pyannote download requires auth."
        )
    return token


async def _drain_with_timeout(url: str, max_s: float, timeout_s: float) -> tuple[np.ndarray, int]:
    """Collect ``max_s`` of PCM with a hard wall-clock budget.

    Wraps the stream in :func:`asyncio.wait_for` so a ffmpeg / yt-dlp
    stall (rate-limit retry, HLS playlist quirk, network blip) raises
    :class:`asyncio.TimeoutError` instead of blocking the suite. The
    ``return`` from the inner generator propagates the close down to
    podcast-helper's ffmpeg child, which terminates promptly.
    """
    chunks: list[np.ndarray] = []
    sample_rate = SR

    async def collect() -> None:
        nonlocal sample_rate
        async for f in _clipped_url_source(url, max_s, realtime=False):
            sample_rate = f["sample_rate"]
            chunks.append(f["pcm"])

    await asyncio.wait_for(collect(), timeout=timeout_s)
    pcm = np.concatenate(chunks, axis=0).astype(np.float32, copy=False) if chunks else np.zeros(0, dtype=np.float32)
    return pcm, sample_rate


@pytest.fixture(scope="module")
def clip_pcm(ph) -> tuple[np.ndarray, int]:
    """Stream the head ``CLIP_S`` seconds of the URL via podcast-helper.

    Returned shape : ``(n_samples,)``, dtype float32, mono 16 kHz.
    Skips on timeout — a stall means the network or yt-dlp / ffmpeg
    is in a degraded state, not that the pipeline is broken.
    """
    try:
        pcm, sample_rate = asyncio.run(
            _drain_with_timeout(VIDEO_URL, CLIP_S, INGEST_TIMEOUT_S)
        )
    except asyncio.TimeoutError:
        pytest.skip(
            f"podcast-helper did not deliver {CLIP_S}s of audio within "
            f"{INGEST_TIMEOUT_S}s ; check ffmpeg / yt-dlp / network."
        )
    if pcm.size == 0:
        pytest.skip("podcast_helper.extract_audio_stream yielded zero frames")
    return pcm, sample_rate


@pytest.fixture(scope="module")
def smoke_pcm(ph) -> tuple[np.ndarray, int]:
    try:
        pcm, sample_rate = asyncio.run(
            _drain_with_timeout(VIDEO_URL, SMOKE_CLIP_S, INGEST_TIMEOUT_S)
        )
    except asyncio.TimeoutError:
        pytest.skip(
            f"podcast-helper did not deliver {SMOKE_CLIP_S}s of audio "
            f"within {INGEST_TIMEOUT_S}s ; check ffmpeg / yt-dlp / network."
        )
    if pcm.size == 0:
        pytest.skip("podcast_helper.extract_audio_stream yielded zero frames")
    return pcm, sample_rate


@pytest.fixture(scope="module")
def yt_subtitles(tmp_path_factory: pytest.TempPathFactory, yh) -> Path:
    """Download YouTube auto-captions, tolerant to per-language failures.

    ``youtube_helper.video_subtitles`` forwards every requested language
    to yt-dlp in a single shot ; a 429 on one language aborts the whole
    call. We try languages one at a time and accept the first success,
    so the test stays usable when YT rate-limits us on a secondary
    language.
    """
    import time

    dest = tmp_path_factory.mktemp("captions")
    last_err: Exception | None = None
    for lang in ("en", "fr"):
        try:
            paths = yh.video_subtitles(
                VIDEO_URL, output_dir=dest, langs=(lang,), auto_only=True,
            )
        except Exception as e:  # noqa: BLE001 — yt-dlp surfaces many error types
            last_err = e
            # Cheap back-off — every retry tickles YT's 429 budget.
            time.sleep(2.0)
            continue
        if paths:
            return Path(next(iter(paths.values())))
    pytest.skip(
        f"no captions retrievable for {VIDEO_URL} "
        f"(last error: {last_err})"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_youtube_captions_fetchable(yt_subtitles: Path) -> None:
    """Sanity : reference exists and has at least 10 words."""
    text = _parse_vtt_text(yt_subtitles)
    assert text, "VTT parsed to empty string"
    words = _normalise_words(text)
    assert len(words) >= 10, f"only {len(words)} unique words in captions"


@pytest.mark.integration
def test_streaming_pipeline_yields_utterance(
    smoke_pcm: tuple[np.ndarray, int], hf_token: str,
) -> None:
    """Live :class:`vh.Pipeline` surfaces at least one non-empty utterance.

    The fixture has already streamed audio via podcast-helper — we feed
    the in-memory buffer into the streaming Pipeline. This isolates the
    pipeline assertion from network flakiness while still exercising
    Silero VAD → online diar → whisper.cpp end-to-end.
    """
    pcm, sample_rate = smoke_pcm
    pipeline = vh.Pipeline(
        source=lambda: vh.sources.from_numpy_array(pcm, sample_rate=sample_rate),
        config=vh.PipelineConfig(
            diar={"backend": "pyannote", "hf_token": hf_token},
            asr={"language": "auto"},
        ),
    )

    async def collect() -> list:
        return [
            ev async for ev in pipeline.run()
        ]

    try:
        events = asyncio.run(asyncio.wait_for(collect(), PIPELINE_TIMEOUT_S))
    except asyncio.TimeoutError:
        pytest.fail(
            f"streaming Pipeline did not finish {SMOKE_CLIP_S}s of audio "
            f"within {PIPELINE_TIMEOUT_S}s ; likely a model load stall or "
            "downstream deadlock."
        )
    utterances = [e for e in events if "text" in e]
    assert utterances, f"no utterance emitted in {SMOKE_CLIP_S} s of audio"
    assert any(u["text"].strip() for u in utterances), (
        "every utterance text was empty"
    )


@pytest.mark.integration
def test_offline_pipeline_vs_youtube_captions(
    clip_pcm: tuple[np.ndarray, int],
    yt_subtitles: Path,
    hf_token: str,
) -> None:
    """Compare :class:`vh.OfflinePipeline` transcript with YouTube captions.

    Two noisy hypotheses of the same audio (Google STT vs whisper turbo
    on pyannote-segmented chunks). We don't expect identity — we expect
    non-trivial lexical overlap. Jaccard ≥ ``MIN_JACCARD`` is the contract.

    ``device=None`` lets the new ``_auto_torch_device`` helper in
    ``vocal_helper.diar`` pick CUDA > MPS > CPU. On Apple Silicon this
    promotes pyannote 3.1 from ~ 15× real-time (CPU) to roughly real-
    time (MPS) — the test finishes well inside ``PIPELINE_TIMEOUT_S``.
    """
    pcm, sample_rate = clip_pcm
    pipeline = vh.OfflinePipeline(
        source=lambda: vh.sources.from_numpy_array(pcm, sample_rate=sample_rate),
        config=vh.OfflinePipelineConfig(
            diar={"backend": "pyannote", "hf_token": hf_token},
            asr={"language": "auto"},
        ),
    )

    async def collect() -> list:
        return [
            ev async for ev in pipeline.run()
        ]

    try:
        events = asyncio.run(asyncio.wait_for(collect(), PIPELINE_TIMEOUT_S))
    except asyncio.TimeoutError:
        pytest.fail(
            f"OfflinePipeline did not finish {CLIP_S}s of audio within "
            f"{PIPELINE_TIMEOUT_S}s ; likely a pyannote 3.1 stall or "
            "downstream deadlock."
        )
    utterances = [e for e in events if "text" in e]
    transcript = " ".join(u["text"] for u in utterances)
    assert transcript.strip(), "offline pipeline produced empty transcript"

    reference = _parse_vtt_text(yt_subtitles, max_seconds=CLIP_S)
    ours = _normalise_words(transcript)
    theirs = _normalise_words(reference)
    score = _jaccard(ours, theirs)

    # Surface both sides in the failure message — debugging an ASR
    # regression is much easier with the word sets in front of you.
    assert score >= MIN_JACCARD, (
        f"jaccard {score:.3f} < {MIN_JACCARD} ; "
        f"ours={len(ours)} words, theirs={len(theirs)} words, "
        f"shared={len(ours & theirs)} words.\n"
        f"sample ours : {sorted(ours)[:20]}\n"
        f"sample yt   : {sorted(theirs)[:20]}"
    )


@pytest.mark.integration
def test_streaming_realtime_pacing(ph) -> None:
    """``from_url(realtime=True)`` paces at wall-clock, not burst-decode.

    Consume 5 s of audio with realtime pacing on and confirm the wall-
    clock elapsed time is within ``[4.0, 9.0]`` s. The lower bound
    catches a regression where ffmpeg's ``-re`` flag is dropped (then
    we'd burst-decode in < 1 s) ; the upper bound is generous to absorb
    URL resolution and ffmpeg startup latency.
    """
    target_s = 5.0
    t0 = time.monotonic()

    async def drain() -> int:
        n_samples = 0
        async for f in _clipped_url_source(VIDEO_URL, target_s, realtime=True):
            n_samples += f["pcm"].shape[0]
        return n_samples

    n = asyncio.run(drain())
    elapsed = time.monotonic() - t0

    # We asked for ~target_s of audio, so n / sr ≈ target_s.
    audio_s = n / SR
    assert audio_s >= target_s - 0.5, f"only {audio_s:.2f} s yielded"
    assert 4.0 <= elapsed <= 9.0, (
        f"realtime pacing off : elapsed={elapsed:.2f}s for {audio_s:.2f}s "
        "of audio (expected ~5s ± ffmpeg startup)"
    )
