"""WhisperStage — construction + batched-path tests (stubbed model, no whisper.cpp)."""

from __future__ import annotations

import numpy as np

from vocal_helper.asr import (
    DEFAULT_INITIAL_PROMPT,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHUNK_S,
    DEFAULT_MIN_SEGMENT_MS,
    DEFAULT_MODEL,
    DEFAULT_THREADS,
    WhisperStage,
    _assign_window,
    _pack_segments,
)
from vocal_helper.types import DiarizedSegment

SR = 16000


def _seg(t0: float, t1: float, speaker: str) -> DiarizedSegment:
    n = int(round((t1 - t0) * SR))
    return DiarizedSegment(
        t0=t0,
        t1=t1,
        sample_rate=SR,
        speaker=speaker,
        pcm=np.zeros(n, dtype=np.float32),
    )


def test_whisper_defaults() -> None:
    """Defaults reflect the canonical settings without loading whisper.cpp."""
    stage = WhisperStage()
    assert stage.model_name == DEFAULT_MODEL
    assert stage.language == DEFAULT_LANGUAGE
    assert stage.threads == DEFAULT_THREADS
    assert stage.word_timestamps is True
    assert stage.initial_prompt == DEFAULT_INITIAL_PROMPT == ""
    assert stage.min_segment_ms == DEFAULT_MIN_SEGMENT_MS
    # Lazy — model not loaded yet.
    assert stage._model is None


def test_whisper_accepts_overrides() -> None:
    stage = WhisperStage(
        model="base",
        language="fr",
        threads=4,
        word_timestamps=False,
        initial_prompt="meeting transcript: scope, budget, deliverables",
        min_segment_ms=500,
    )
    assert stage.model_name == "base"
    assert stage.language == "fr"
    assert stage.threads == 4
    assert stage.word_timestamps is False
    assert stage.initial_prompt.startswith("meeting transcript")
    assert stage.min_segment_ms == 500


def test_initial_prompt_documented_default_is_empty() -> None:
    """Empty default keeps the zero-config path functional ;
    docstring strongly recommends providing a domain-aligned prompt."""
    assert DEFAULT_INITIAL_PROMPT == ""


# ----- batched offline path -------------------------------------------------


def test_batch_defaults_off() -> None:
    """Batching + warm-up are opt-in ; bare WhisperStage keeps per-segment behaviour."""
    stage = WhisperStage()
    assert stage.batch is False
    assert stage.max_chunk_s == DEFAULT_MAX_CHUNK_S
    assert stage.warmup is False


def test_pipeline_wiring_defaults() -> None:
    """Offline defaults to batched ASR ; streaming defaults to warm-up."""
    import vocal_helper as voh

    src = lambda: iter(())  # noqa: E731 — never run, just to construct.
    online = voh.Pipeline(source=src)
    assert online._asr.batch is False
    assert online._asr.warmup is True

    offline = voh.OfflinePipeline(source=src)
    assert offline._asr.batch is True
    assert offline._asr.warmup is False

    # Explicit config always wins over the pipeline default.
    off2 = voh.OfflinePipeline(source=src, config=voh.OfflinePipelineConfig(asr={"batch": False}))
    assert off2._asr.batch is False


def test_pack_segments_respects_chunk_cap() -> None:
    segs = [_seg(0, 2, "S0"), _seg(2, 4, "S1"), _seg(4, 6, "S0"), _seg(6, 20, "S1")]
    # cap 5 s : [2+2] together (4 + pad ≤ 5), then the 2 s, then the 14 s alone.
    chunks = _pack_segments(segs, max_chunk_s=5.0)
    assert [len(c) for c in chunks] == [2, 1, 1]
    # cap huge : everything in one chunk.
    assert len(_pack_segments(segs, max_chunk_s=1000.0)) == 1
    # cap tiny : every segment isolated.
    assert [len(c) for c in _pack_segments(segs, max_chunk_s=0.5)] == [1, 1, 1, 1]


def test_assign_window_containment_and_nearest() -> None:
    windows = [(0.0, 2.0), (2.1, 4.1), (4.2, 6.2)]
    assert _assign_window(windows, 1.0) == 0
    assert _assign_window(windows, 3.0) == 1
    assert _assign_window(windows, 6.0) == 2
    # In a pad gap → nearest by edge distance.
    assert _assign_window(windows, 2.05) in (0, 1)
    # Past the end → last window.
    assert _assign_window(windows, 99.0) == 2


class _Phrase:
    def __init__(self, text: str, t0_cs: float, t1_cs: float, language: str | None = None):
        self.text = text
        self.t0 = t0_cs  # centiseconds, chunk-local (whisper.cpp convention)
        self.t1 = t1_cs
        self.language = language


class _StubModel:
    """Returns phrases positioned to fall inside specific segment windows."""

    def __init__(self, phrases: list[_Phrase]):
        self._phrases = phrases

    def transcribe(self, pcm, **kwargs):  # noqa: ANN001
        return self._phrases


def test_batched_chunk_preserves_one_utterance_per_segment() -> None:
    # Three segments: S0 [0,2], S1 [2,3], S0 [3,5]. Concatenated with 0.1 s pads,
    # local windows ≈ [0,2] [2.1,3.1] [3.2,5.2].
    chunk = [_seg(0, 2, "S0"), _seg(2, 3, "S1"), _seg(3, 5, "S0")]
    stage = WhisperStage(batch=True)
    stage._model = _StubModel(
        [
            _Phrase("hello", 50, 150, "en"),  # mid 1.0 → window 0 (S0)
            _Phrase("bonjour", 250, 300, "fr"),  # mid 2.75 → window 1 (S1)
            _Phrase("world", 340, 500, "en"),  # mid 4.2 → window 2 (S0)
        ]
    )
    utts = stage._transcribe_chunk_blocking(chunk)
    # Contract: exactly one Utterance per input segment, in order, with diar speaker.
    assert len(utts) == 3
    assert [u["speaker"] for u in utts] == ["S0", "S1", "S0"]
    assert [u["t0"] for u in utts] == [0, 2, 3]
    assert utts[0]["text"] == "hello"
    assert utts[1]["text"] == "bonjour"
    assert utts[2]["text"] == "world"
    # Word abs-time is mapped back into the owning segment's real timeline.
    assert utts[1]["language"] == "fr"
    w_t0, w_t1, w_text = utts[1]["words"][0]
    assert w_text == "bonjour"
    assert 2.0 <= w_t0 <= 3.0  # inside S1's [2,3]


def test_batched_chunk_transcribe_failure_keeps_contract() -> None:
    class _Boom:
        def transcribe(self, pcm, **kwargs):  # noqa: ANN001
            raise RuntimeError("whisper exploded")

    chunk = [_seg(0, 2, "S0"), _seg(2, 4, "S1")]
    stage = WhisperStage(batch=True)
    stage._model = _Boom()
    utts = stage._transcribe_chunk_blocking(chunk)
    assert [u["speaker"] for u in utts] == ["S0", "S1"]
    assert all(u["text"] == "" for u in utts)
