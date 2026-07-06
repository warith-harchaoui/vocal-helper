"""WhisperStage — construction-time tests only (no model load)."""
from __future__ import annotations

from vocal_helper.asr import (
    DEFAULT_INITIAL_PROMPT,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_SEGMENT_MS,
    DEFAULT_MODEL,
    DEFAULT_THREADS,
    WhisperStage,
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
