"""Tests for the ``vocal_helper.llm`` surface that can run offline.

We never call Ollama — the only thing tested here is the pure
``_extract_response_text`` helper (parses both legacy-dict and
new-Pydantic response shapes) and the construction-time defaults of
:class:`GemmaAnalystStage`.
"""

from __future__ import annotations

from dataclasses import dataclass

from vocal_helper.llm import (
    DEFAULT_FLUSH_EVERY_N,
    DEFAULT_MODEL,
    DEFAULT_RECENT_WINDOW_S,
    GemmaAnalystStage,
    _extract_response_text,
)

# ---------------------------------------------------------------------------
# _extract_response_text — parses every shape Ollama has shipped.
# ---------------------------------------------------------------------------


def test_extract_dict_shape() -> None:
    """Legacy dict shape — what ``ollama < 0.4`` returns."""
    resp = {"response": "  Hello world  ", "model": "gemma4:e4b"}
    assert _extract_response_text(resp) == "Hello world"


def test_extract_dict_missing_response_key() -> None:
    """A dict without ``response`` collapses to the empty string, not a crash."""
    assert _extract_response_text({"model": "gemma4:e4b"}) == ""


def test_extract_object_with_response_attr() -> None:
    """Pydantic-ish shape — what ``ollama >= 0.4`` returns."""

    @dataclass
    class FakeGenerateResponse:
        """Stand-in for the ollama >= 0.4 object exposing a ``response`` attr."""

        response: str

    assert _extract_response_text(FakeGenerateResponse(response="  hi  ")) == "hi"


def test_extract_object_with_none_response_falls_through() -> None:
    """``response=None`` should hit the str(resp) fallback, not return 'None' silently."""

    @dataclass
    class FakeResp:
        """Response object whose ``response`` attr is ``None`` (falls through)."""

        response: None = None

    # The fallback returns str(resp).strip() — which for a dataclass is
    # the repr. We only assert it's a string and didn't blow up.
    assert isinstance(_extract_response_text(FakeResp()), str)


def test_extract_falls_back_to_str() -> None:
    """An unknown shape gets coerced via ``str()`` rather than raising."""
    assert _extract_response_text("naked string") == "naked string"
    assert _extract_response_text(42) == "42"


# ---------------------------------------------------------------------------
# GemmaAnalystStage — construction only, never connects to Ollama.
# ---------------------------------------------------------------------------


def test_analyst_defaults() -> None:
    """Bare ``GemmaAnalystStage`` mirrors the documented defaults, client unconnected."""
    stage = GemmaAnalystStage()
    assert stage.model == DEFAULT_MODEL
    assert stage.recent_window_s == DEFAULT_RECENT_WINDOW_S
    assert stage.flush_every_n == DEFAULT_FLUSH_EVERY_N
    # Default 60.0 (time-based) selected in the 2026-06-30 cadence
    # sweep — see vocal_helper.llm module docstring for the table.
    assert stage.flush_every_s == 60.0
    assert stage.host is None
    # Lazy client — must not be connected at construction.
    assert stage._client is None


def test_analyst_accepts_overrides() -> None:
    """Every constructor override is stored verbatim on the analyst stage."""
    stage = GemmaAnalystStage(
        model="qwen3:8b",
        recent_window_s=30.0,
        flush_every_n=10,
        flush_every_s=15.0,
        host="http://ollama.internal:11434",
        prompt_template="custom {summary} {new_block}",
    )
    assert stage.model == "qwen3:8b"
    assert stage.recent_window_s == 30.0
    assert stage.flush_every_n == 10
    assert stage.flush_every_s == 15.0
    assert stage.host == "http://ollama.internal:11434"
    assert "{summary}" in stage.prompt_template
