"""
vocal_helper.types
==================

Typed dicts and dataclasses passed between pipeline stages.

The shapes are deliberately small and JSON-friendly so the same
events can be (a) consumed by an in-process subscriber, (b) shipped
over a WebSocket / SSE feed, (c) stored as JSONL for replay.

PCM convention — the same one used across the AI Helpers suite
(``capture_helper.MicFrame``, ``podcast_helper.PcmFrame``) :

- mono ``np.float32``
- 16 kHz (resampling is the source's responsibility ; the pipeline
  assumes the configured ``sample_rate``).

Time convention :

- ``t0`` / ``t1`` are seconds since pipeline ``start_at`` (the
  monotonic time the pipeline began consuming frames). All stages
  use the same clock so events can be aligned downstream.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class PcmFrame(TypedDict):
    """One PCM frame at the configured sample rate.

    Mirrors :class:`capture_helper.MicFrame` and
    :class:`podcast_helper.PcmFrame` so a producer from either
    library can feed this pipeline directly.
    """

    t0: float           # seconds since pipeline start
    sample_rate: int    # Hz
    pcm: NDArray[np.float32]  # shape (n_samples,), mono float32


class VoicedSegment(TypedDict):
    """A contiguous run of voiced speech as detected by VAD.

    Emitted by :class:`vocal_helper.vad.SileroVADStage` when speech
    *ends* (i.e. after ``min_silence_ms`` of trailing silence) — the
    PCM buffer holds the full voiced span minus the trailing silence.
    """

    t0: float
    t1: float
    sample_rate: int
    pcm: NDArray[np.float32]


class DiarizedSegment(TypedDict):
    """Voiced segment with a global speaker id attached.

    Emitted by :class:`vocal_helper.diar.OnlineDiarStage` after each
    voiced segment has been embedded and matched against the running
    speaker centroids. ``speaker`` is a stable string id of the form
    ``"S0"``, ``"S1"`` — same speaker across the whole session.
    """

    t0: float
    t1: float
    sample_rate: int
    speaker: str
    pcm: NDArray[np.float32]


class Utterance(TypedDict):
    """Transcribed diarized segment.

    Emitted by :class:`vocal_helper.asr.WhisperStage`. ``words`` is
    a list of ``(t0, t1, text)`` triplets when the underlying
    backend supports word-level timestamps, else a single triplet
    spanning the whole utterance.
    """

    t0: float
    t1: float
    speaker: str
    text: str
    words: list[tuple[float, float, str]]
    language: str | None


class SummarySnapshot(TypedDict):
    """One running summary as produced by the optional LLM analyst.

    ``recent`` is the verbatim transcript of the last
    ``recent_window_s`` seconds ; ``summary`` is the LLM's running
    digest of everything older than that.
    """

    t0: float            # snapshot time (= newest utterance's t1)
    summary: str         # rolling digest, older than recent window
    recent: str          # verbatim recent transcript
    model: str           # ollama model name (e.g. "gemma3:4b")
