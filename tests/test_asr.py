"""WhisperStage — construction, batched offline path, and one-shot helpers.

All tests are model-free: the whisper.cpp model is stubbed or replaced so no
real backend is loaded. They cover the construction contract (defaults +
overrides + pipeline wiring), the batched decode's "one utterance per segment"
invariant (success and failure), the packing/windowing helpers that feed it,
and the discovery-first one-shot helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

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
    transcribe_pcm,
    transcribe_pcm_with_language,
)
from vocal_helper.types import DiarizedSegment, Utterance

SR = 16000


def _seg(t0: float, t1: float, speaker: str) -> DiarizedSegment:
    """Build a silent :class:`DiarizedSegment` spanning ``[t0, t1]`` for one speaker."""
    n = int(round((t1 - t0) * SR))
    return DiarizedSegment(
        t0=t0,
        t1=t1,
        sample_rate=SR,
        speaker=speaker,
        pcm=np.zeros(n, dtype=np.float32),
    )


# ----- construction contract ------------------------------------------------


def test_whisper_construction_contract() -> None:
    """Defaults are canonical and lazy; every constructor override is stored verbatim.

    Merges the defaults, override, empty-prompt, and batch-off checks into one
    construction contract: a bare stage carries the documented defaults (and
    hasn't loaded a model), while an explicitly-configured stage echoes each
    field back unchanged.
    """
    stage = WhisperStage()
    assert stage.model_name == DEFAULT_MODEL
    assert stage.language == DEFAULT_LANGUAGE
    assert stage.threads == DEFAULT_THREADS
    assert stage.word_timestamps is True
    # Empty default keeps the zero-config path working; the docstring still
    # recommends a domain-aligned prompt.
    assert stage.initial_prompt == DEFAULT_INITIAL_PROMPT == ""
    assert stage.min_segment_ms == DEFAULT_MIN_SEGMENT_MS
    assert stage._model is None  # lazy — model not loaded until first use
    # Batching + warm-up are opt-in; a bare stage keeps per-segment behaviour.
    assert stage.batch is False
    assert stage.max_chunk_s == DEFAULT_MAX_CHUNK_S
    assert stage.warmup is False

    # Every override is stored as given.
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


def test_pipeline_wiring_defaults() -> None:
    """Offline defaults to batched ASR; streaming defaults to warm-up; config wins."""
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


# ----- packing + windowing helpers ------------------------------------------


def test_pack_and_window_helpers() -> None:
    """``_pack_segments`` respects the chunk cap and ``_assign_window`` maps times.

    Two helpers that feed the batched decode, checked together: packing groups
    segments up to ``max_chunk_s`` (isolating over-cap ones, collapsing under a
    huge cap, splitting under a tiny one), and window assignment resolves a time
    to its containing window or the nearest edge otherwise.
    """
    segs = [_seg(0, 2, "S0"), _seg(2, 4, "S1"), _seg(4, 6, "S0"), _seg(6, 20, "S1")]
    # cap 5 s : [2+2] together (4 + pad ≤ 5), then the 2 s, then the 14 s alone.
    assert [len(c) for c in _pack_segments(segs, max_chunk_s=5.0)] == [2, 1, 1]
    # cap huge : everything in one chunk.
    assert len(_pack_segments(segs, max_chunk_s=1000.0)) == 1
    # cap tiny : every segment isolated.
    assert [len(c) for c in _pack_segments(segs, max_chunk_s=0.5)] == [1, 1, 1, 1]

    windows = [(0.0, 2.0), (2.1, 4.1), (4.2, 6.2)]
    assert _assign_window(windows, 1.0) == 0
    assert _assign_window(windows, 3.0) == 1
    assert _assign_window(windows, 6.0) == 2
    assert _assign_window(windows, 2.05) in (0, 1)  # in a pad gap → nearest edge
    assert _assign_window(windows, 99.0) == 2  # past the end → last window


# ----- batched offline decode -----------------------------------------------


class _Phrase:
    """Minimal stand-in for a whisper.cpp phrase (text + centisecond timings)."""

    def __init__(self, text: str, t0_cs: float, t1_cs: float, language: str | None = None):
        """Store the phrase text, chunk-local timings, and optional language tag."""
        self.text = text
        self.t0 = t0_cs  # centiseconds, chunk-local (whisper.cpp convention)
        self.t1 = t1_cs
        self.language = language


class _StubModel:
    """Returns phrases positioned to fall inside specific segment windows."""

    def __init__(self, phrases: list[_Phrase]):
        """Capture the canned phrases to replay on every ``transcribe`` call."""
        self._phrases = phrases

    def transcribe(self, pcm, **kwargs):  # noqa: ANN001
        """Ignore the audio and return the pre-positioned phrases."""
        return self._phrases


def test_batched_chunk_one_utterance_per_segment() -> None:
    """Batched decode emits exactly one utterance per segment — on success and on crash.

    The invariant is the same in both paths: N input segments → N utterances,
    in order, each tagged with its diarization speaker. On success, phrases are
    routed to their owning segment and word times are mapped back to the real
    timeline; on a model crash, each segment still yields an empty utterance
    (never fewer).
    """
    # Success: three segments S0 [0,2], S1 [2,3], S0 [3,5]. Concatenated with
    # 0.1 s pads, local windows ≈ [0,2] [2.1,3.1] [3.2,5.2].
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
    # Exactly one Utterance per input segment, in order, carrying the diar speaker.
    assert len(utts) == 3
    assert [u["speaker"] for u in utts] == ["S0", "S1", "S0"]
    assert [u["t0"] for u in utts] == [0, 2, 3]
    assert [u["text"] for u in utts] == ["hello", "bonjour", "world"]
    assert utts[1]["language"] == "fr"
    # Word abs-time is mapped back into the owning segment's real timeline.
    w_t0, _w_t1, w_text = utts[1]["words"][0]
    assert w_text == "bonjour"
    assert 2.0 <= w_t0 <= 3.0  # inside S1's [2,3]

    # Failure: a model that always raises still preserves one empty utterance
    # per segment, so downstream indexing never desyncs.
    class _Boom:
        """A model whose ``transcribe`` always raises, to test the failure path."""

        def transcribe(self, pcm, **kwargs):  # noqa: ANN001
            """Simulate a whisper.cpp blow-up mid-decode."""
            raise RuntimeError("whisper exploded")

    stage = WhisperStage(batch=True)
    stage._model = _Boom()
    utts = stage._transcribe_chunk_blocking([_seg(0, 2, "S0"), _seg(2, 4, "S1")])
    assert [u["speaker"] for u in utts] == ["S0", "S1"]
    assert all(u["text"] == "" for u in utts)


# ----- transcribe_pcm / _with_language (discovery-first, model-free) ---------


def _stub_whisper(monkeypatch: pytest.MonkeyPatch, text: str, language: str | None) -> None:
    """Patch :class:`WhisperStage` so no real whisper.cpp model is loaded.

    ``_ensure_model`` becomes a no-op and ``_transcribe_blocking`` returns a
    canned :class:`Utterance` carrying ``text`` and the *discovered* ``language``
    — exactly what the one-shot helpers read back.
    """
    # No model load: the helpers call _ensure_model() before transcribing.
    monkeypatch.setattr(WhisperStage, "_ensure_model", lambda self: None)

    def _fake_transcribe(self: WhisperStage, seg: DiarizedSegment) -> Utterance:
        """Return a fixed utterance, echoing the segment's timing/speaker."""
        return Utterance(
            t0=seg["t0"],
            t1=seg["t1"],
            speaker=seg["speaker"],
            text=text,
            words=[],
            language=language,
        )

    monkeypatch.setattr(WhisperStage, "_transcribe_blocking", _fake_transcribe)


def test_one_shot_helpers_surface_discovered_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helpers return whisper's *discovered* language, and ``transcribe_pcm``
    keeps its text-only contract.

    ``transcribe_pcm_with_language`` surfaces the language whisper actually found
    (not the "auto" the caller passed), while ``transcribe_pcm`` delegates to it
    and hands back only the text string.
    """
    # whisper "discovers" French even though the caller left language on "auto".
    _stub_whisper(monkeypatch, "bonjour", "fr")
    text, language = transcribe_pcm_with_language(np.zeros(SR, dtype=np.float32), sr=SR)
    assert text == "bonjour"
    assert language == "fr"

    # transcribe_pcm delegates to the language helper but returns only the text.
    _stub_whisper(monkeypatch, "hola", "es")
    out = transcribe_pcm(np.zeros(SR, dtype=np.float32), sr=SR)
    assert out == "hola"
    assert isinstance(out, str)
