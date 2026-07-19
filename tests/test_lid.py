"""Acoustic LID — region coalescing + short-region absorption (no whisper.cpp).

The model-backed path (:func:`detect_language` /
:func:`detect_language_regions`'s per-window inference) needs whisper.cpp
and lives behind the ``integration`` marker. The region-shaping logic —
:func:`_coalesce` and :func:`_absorb_short_regions` — is pure and fully
testable on synthetic :class:`LangRegion` lists, which is where the
correctness of the multi-language partitioning actually lives.
"""

from __future__ import annotations

import numpy as np
import pytest

import vocal_helper.lid as lid
from vocal_helper.lid import (
    DEFAULT_FAST_CONF_GATE,
    DEFAULT_MIN_REGION_S,
    DEFAULT_SUPPORTED_LANGS,
    DEFAULT_WINDOW_S,
    LangRegion,
    _absorb_short_regions,
    _coalesce,
    _snap_boundaries_to_silence,
    detect_language_regions_fast,
)


def _spans(regions: list[LangRegion]) -> list[tuple[str, float, float]]:
    """Flatten regions to ``(lang, t0, t1)`` triples for terse comparisons."""
    return [(r.lang, r.t0, r.t1) for r in regions]


# ----- defaults -------------------------------------------------------------


def test_lid_defaults() -> None:
    """LID default constants match the documented window / min-region / lang set."""
    assert DEFAULT_WINDOW_S == 10.0
    assert DEFAULT_MIN_REGION_S == 8.0
    # DEFAULT_SUPPORTED_LANGS is an opt-in routing hint, never a default: no
    # single language is privileged, so we assert membership as a set (order
    # carries no meaning) rather than which code happens to come first.
    assert {"en", "fr", "es", "it", "pl", "nl"} <= set(DEFAULT_SUPPORTED_LANGS)
    # Canonical ISO-639-1 : two-letter, lower-case, no region suffixes.
    assert all(len(c) == 2 and c.islower() for c in DEFAULT_SUPPORTED_LANGS)


def test_snap_boundary_moves_to_silence() -> None:
    """A language boundary snaps into the nearby silent gap, not mid-speech."""
    # 20 s tone with a silent gap at 9.5-10.5 s; a boundary at 9.0 s must snap
    # into the gap. Pure — no whisper model.
    sr = 16_000
    t = np.arange(20 * sr) / sr
    pcm = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    pcm[int(9.5 * sr) : int(10.5 * sr)] = 0.0
    regions = [LangRegion("fr", 0, 9.0), LangRegion("es", 9.0, 20.0)]
    out = _snap_boundaries_to_silence(pcm, sr, regions, snap_s=1.5)
    assert 9.5 <= out[0].t1 <= 10.5
    assert out[1].t0 == out[0].t1


# ----- _coalesce ------------------------------------------------------------


def test_coalesce_merges_same_language_neighbours() -> None:
    """Adjacent same-language regions collapse into one span."""
    regions = [
        LangRegion("en", 0.0, 20.0),
        LangRegion("en", 20.0, 40.0),
        LangRegion("fr", 40.0, 60.0),
        LangRegion("fr", 60.0, 80.0),
        LangRegion("en", 80.0, 100.0),
    ]
    assert _spans(_coalesce(regions)) == [
        ("en", 0.0, 40.0),
        ("fr", 40.0, 80.0),
        ("en", 80.0, 100.0),
    ]


def test_coalesce_empty_and_singleton() -> None:
    """Empty and single-region inputs pass through ``_coalesce`` unchanged."""
    assert _coalesce([]) == []
    one = [LangRegion("fr", 0.0, 30.0)]
    assert _spans(_coalesce(one)) == [("fr", 0.0, 30.0)]


def test_coalesce_no_adjacent_duplicates_is_identity() -> None:
    """With no same-language neighbours, ``_coalesce`` is a no-op."""
    regions = [
        LangRegion("en", 0.0, 20.0),
        LangRegion("fr", 20.0, 40.0),
        LangRegion("en", 40.0, 60.0),
    ]
    assert _spans(_coalesce(regions)) == _spans(regions)


# ----- _absorb_short_regions ------------------------------------------------


def test_absorb_short_region_into_longer_neighbour() -> None:
    """A short blip between two long same-language stretches is absorbed and merged."""
    # A 4 s "fr" blip between two long "en" stretches is whisper noise :
    # relabel it "en" and re-coalesce back to a single region.
    regions = [
        LangRegion("en", 0.0, 30.0),
        LangRegion("fr", 30.0, 34.0),
        LangRegion("en", 34.0, 60.0),
    ]
    absorbed = _absorb_short_regions(regions, min_region_s=DEFAULT_MIN_REGION_S)
    assert _spans(absorbed) == [("en", 0.0, 60.0)]


def test_absorb_keeps_real_switch() -> None:
    """Two regions that both clear the threshold are a real switch, left intact."""
    # Both regions clear the threshold → a genuine switch, left untouched.
    regions = [
        LangRegion("en", 0.0, 30.0),
        LangRegion("fr", 30.0, 60.0),
    ]
    absorbed = _absorb_short_regions(regions, min_region_s=DEFAULT_MIN_REGION_S)
    assert _spans(absorbed) == _spans(regions)


def test_absorb_short_region_prefers_longer_side() -> None:
    """A short region is relabelled to its longer neighbour, not just the left one."""
    # The short "es" region sits between a 25 s "en" and a 12 s "fr" — it is
    # relabelled to the *longer* (en) neighbour, not merely the left one.
    regions = [
        LangRegion("en", 0.0, 25.0),
        LangRegion("es", 25.0, 30.0),
        LangRegion("fr", 30.0, 42.0),
    ]
    absorbed = _absorb_short_regions(regions, min_region_s=10.0)
    assert _spans(absorbed) == [("en", 0.0, 30.0), ("fr", 30.0, 42.0)]


def test_absorb_short_region_uses_right_when_no_left() -> None:
    """A leading short region with no left neighbour is absorbed rightward."""
    # Leading short region has no left neighbour → absorbed rightward.
    regions = [
        LangRegion("fr", 0.0, 3.0),
        LangRegion("en", 3.0, 40.0),
    ]
    absorbed = _absorb_short_regions(regions, min_region_s=10.0)
    assert _spans(absorbed) == [("en", 0.0, 40.0)]


def test_absorb_single_region_untouched_even_if_short() -> None:
    """A lone short region survives — a short monolingual file mustn't vanish."""
    # A lone region is always returned — a short monolingual file must not
    # collapse to nothing.
    one = [LangRegion("nl", 0.0, 2.0)]
    assert _spans(_absorb_short_regions(one, min_region_s=10.0)) == [("nl", 0.0, 2.0)]


def test_absorb_cascades_until_all_clear() -> None:
    """Absorption repeats until every remaining region clears the threshold."""
    # Several short blips collapse into the dominant surrounding language.
    regions = [
        LangRegion("en", 0.0, 40.0),
        LangRegion("fr", 40.0, 43.0),
        LangRegion("es", 43.0, 46.0),
        LangRegion("en", 46.0, 90.0),
    ]
    absorbed = _absorb_short_regions(regions, min_region_s=10.0)
    assert _spans(absorbed) == [("en", 0.0, 90.0)]


# ----- detect_language_regions_fast (single-pass fast path) -----------------
#
# The whisper-backed detection is monkeypatched, so these exercise the routing
# logic — confident → one region, uncertain → fall back to the posterior scan —
# without loading a model.


def test_fast_confident_returns_single_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """A confident, routable detection short-circuits to one whole-file region."""
    monkeypatch.setattr(lid, "detect_language", lambda pcm, **kw: ("en", 0.99))
    # The posterior scan must NOT run on the fast path — make it explode if it does.
    monkeypatch.setattr(
        lid,
        "detect_language_regions",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )
    pcm = np.zeros(16_000 * 5, dtype=np.float32)  # 5 s
    out = detect_language_regions_fast(pcm, 16_000)
    assert [(r.lang, r.t0, r.t1) for r in out] == [("en", 0.0, 5.0)]


def test_fast_lowconf_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below the confidence gate, the robust multi-window scan runs instead."""
    sentinel = [LangRegion("fr", 0.0, 3.0), LangRegion("es", 3.0, 6.0)]
    monkeypatch.setattr(lid, "detect_language", lambda pcm, **kw: ("en", 0.2))
    monkeypatch.setattr(lid, "detect_language_regions", lambda *a, **k: sentinel)
    pcm = np.zeros(16_000 * 6, dtype=np.float32)
    out = detect_language_regions_fast(pcm, 16_000, conf_gate=DEFAULT_FAST_CONF_GATE)
    assert out is sentinel


def test_fast_unsupported_language_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A confident but un-routable language still defers to the full scan."""
    sentinel = [LangRegion("en", 0.0, 4.0)]
    monkeypatch.setattr(lid, "detect_language", lambda pcm, **kw: ("zz", 0.99))
    monkeypatch.setattr(lid, "detect_language_regions", lambda *a, **k: sentinel)
    pcm = np.zeros(16_000 * 4, dtype=np.float32)
    out = detect_language_regions_fast(pcm, 16_000, supported=("en", "fr"))
    assert out is sentinel


# ----- detect_language discovery vs opt-in routing (no whisper.cpp) ----------
#
# The whisper stage is faked so we can pin the *policy* — discover freely by
# default, restrict only when asked — without loading a model.


class _FakeModel:
    """Stand-in for pywhispercpp's ``Model`` exposing ``auto_detect_language``."""

    def __init__(self, top: str, prob: float, dist: dict[str, float]) -> None:
        # ``top``/``prob`` = whisper's own argmax ; ``dist`` = full posterior.
        self._top = top
        self._prob = prob
        self._dist = dist

    def auto_detect_language(
        self, pcm: np.ndarray, offset_ms: int = 0
    ) -> tuple[tuple[str, float], dict[str, float]]:
        """Return ``((argmax, prob), full_distribution)`` like whisper.cpp does."""
        return (self._top, self._prob), self._dist


class _FakeStage:
    """Minimal stand-in for the cached :class:`WhisperStage` used by LID."""

    def __init__(self, model: _FakeModel) -> None:
        self._model = model


def test_detect_language_discovers_true_argmax(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no whitelist, detection returns whisper's true top language."""
    # whisper's argmax is Galician (``gl``) — a language NOT in the routing
    # hint. Pure discovery must surface it verbatim, never coerce it away.
    fake = _FakeStage(_FakeModel("gl", 0.82, {"gl": 0.82, "es": 0.1, "en": 0.05}))
    monkeypatch.setattr(lid, "_get_stage", lambda model, threads: fake)
    code, prob = lid.detect_language(np.zeros(16_000, dtype=np.float32))
    assert code == "gl"
    assert prob == pytest.approx(0.82)


def test_detect_language_opt_in_whitelist_reranks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-supplied ``supported`` set re-ranks strictly within it."""
    # Same distribution, but the caller can only route en/es → the guard picks
    # the best routable code (es), dropping the un-routable Galician argmax.
    fake = _FakeStage(_FakeModel("gl", 0.82, {"gl": 0.82, "es": 0.1, "en": 0.05}))
    monkeypatch.setattr(lid, "_get_stage", lambda model, threads: fake)
    code, prob = lid.detect_language(np.zeros(16_000, dtype=np.float32), supported=("en", "es"))
    assert code == "es"
    assert prob == pytest.approx(0.1)


def test_fast_empty_input_returns_no_regions() -> None:
    """Empty audio has no language to discover, so no region is invented."""
    # Discovery, not defaulting: silence must not be labelled a language.
    # Both the fast path and the slow path return an empty list here.
    out = detect_language_regions_fast(np.zeros(0, dtype=np.float32), 16_000)
    assert out == []
