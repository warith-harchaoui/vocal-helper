"""Acoustic LID — discovery policy + region shaping (no whisper.cpp model loads).

The model-backed inference lives behind the ``integration`` marker. Everything
here is either pure region math (:func:`_coalesce`, :func:`_absorb_short_regions`,
:func:`_snap_boundaries_to_silence`) or the routing/discovery *policy* around the
whisper stage, which is monkeypatched. That is where the correctness of the
multi-language partitioning and the discovery-first defaults actually lives.
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


# ----- defaults + silence-aware boundary snapping ---------------------------


def test_lid_defaults_and_silence_snap() -> None:
    """Documented constants hold and a boundary snaps into a nearby silent gap.

    Two coherent facts about the *shaping* layer: the published defaults
    (window / min-region / routing hint), and that a mislocated language
    boundary is nudged into speech-free audio rather than left mid-word. Both
    are pure — no whisper model is loaded.
    """
    # Published windowing / min-region constants.
    assert DEFAULT_WINDOW_S == 10.0
    assert DEFAULT_MIN_REGION_S == 8.0
    # DEFAULT_SUPPORTED_LANGS is an opt-in routing hint, never a default: no
    # single language is privileged, so membership (order-free) is what matters.
    assert {"en", "fr", "es", "it", "pl", "nl"} <= set(DEFAULT_SUPPORTED_LANGS)
    # Canonical ISO-639-1 : two-letter, lower-case, no region suffixes.
    assert all(len(c) == 2 and c.islower() for c in DEFAULT_SUPPORTED_LANGS)

    # 20 s tone with a silent gap at 9.5-10.5 s; a boundary at 9.0 s must snap
    # into the gap so the language switch lands on silence, not speech.
    sr = 16_000
    t = np.arange(20 * sr) / sr
    pcm = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    pcm[int(9.5 * sr) : int(10.5 * sr)] = 0.0
    regions = [LangRegion("fr", 0, 9.0), LangRegion("es", 9.0, 20.0)]
    out = _snap_boundaries_to_silence(pcm, sr, regions, snap_s=1.5)
    assert 9.5 <= out[0].t1 <= 10.5  # boundary moved into the silent gap
    assert out[1].t0 == out[0].t1  # neighbours stay contiguous after the snap


# ----- _coalesce ------------------------------------------------------------


def test_coalesce_merges_neighbours_and_is_otherwise_identity() -> None:
    """``_coalesce`` collapses adjacent same-language spans and touches nothing else.

    Covers the three coalescing outcomes at once: same-language neighbours
    merge, empty/singleton inputs pass through, and an already-alternating list
    is returned unchanged.
    """
    # Adjacent same-language runs collapse into one span each.
    merged = [
        LangRegion("en", 0.0, 20.0),
        LangRegion("en", 20.0, 40.0),
        LangRegion("fr", 40.0, 60.0),
        LangRegion("fr", 60.0, 80.0),
        LangRegion("en", 80.0, 100.0),
    ]
    assert _spans(_coalesce(merged)) == [
        ("en", 0.0, 40.0),
        ("fr", 40.0, 80.0),
        ("en", 80.0, 100.0),
    ]

    # Empty and single-region inputs are pass-through.
    assert _coalesce([]) == []
    one = [LangRegion("fr", 0.0, 30.0)]
    assert _spans(_coalesce(one)) == [("fr", 0.0, 30.0)]

    # With no same-language neighbours, coalescing is a no-op.
    alternating = [
        LangRegion("en", 0.0, 20.0),
        LangRegion("fr", 20.0, 40.0),
        LangRegion("en", 40.0, 60.0),
    ]
    assert _spans(_coalesce(alternating)) == _spans(alternating)


# ----- _absorb_short_regions ------------------------------------------------


def test_absorb_short_regions_policy() -> None:
    """Short regions are absorbed into their longer neighbour; real switches and
    lone regions survive.

    One contract for the whole absorption policy, each case a distinct branch:
    sub-threshold blips get relabelled toward the longer side (left, right, or
    cascading), while regions that clear the threshold — or a single region of
    any length — are preserved untouched.
    """
    # A 4 s "fr" blip between two long "en" stretches is whisper noise: relabel
    # it and re-coalesce back to a single "en" span.
    assert _spans(
        _absorb_short_regions(
            [
                LangRegion("en", 0.0, 30.0),
                LangRegion("fr", 30.0, 34.0),
                LangRegion("en", 34.0, 60.0),
            ],
            min_region_s=DEFAULT_MIN_REGION_S,
        )
    ) == [("en", 0.0, 60.0)]

    # Both regions clear the threshold → a genuine switch, left untouched.
    real_switch = [LangRegion("en", 0.0, 30.0), LangRegion("fr", 30.0, 60.0)]
    assert _spans(_absorb_short_regions(real_switch, min_region_s=DEFAULT_MIN_REGION_S)) == [
        ("en", 0.0, 30.0),
        ("fr", 30.0, 60.0),
    ]

    # Short "es" between a 25 s "en" and a 12 s "fr": relabel to the *longer*
    # (en) neighbour, not merely the left one.
    assert _spans(
        _absorb_short_regions(
            [
                LangRegion("en", 0.0, 25.0),
                LangRegion("es", 25.0, 30.0),
                LangRegion("fr", 30.0, 42.0),
            ],
            min_region_s=10.0,
        )
    ) == [("en", 0.0, 30.0), ("fr", 30.0, 42.0)]

    # Leading short region has no left neighbour → absorbed rightward.
    assert _spans(
        _absorb_short_regions(
            [LangRegion("fr", 0.0, 3.0), LangRegion("en", 3.0, 40.0)],
            min_region_s=10.0,
        )
    ) == [("en", 0.0, 40.0)]

    # A lone short region survives — a short monolingual file must not collapse
    # to nothing.
    assert _spans(_absorb_short_regions([LangRegion("nl", 0.0, 2.0)], min_region_s=10.0)) == [
        ("nl", 0.0, 2.0)
    ]

    # Several consecutive short blips cascade into the dominant surroundings
    # until every remaining region clears the threshold.
    assert _spans(
        _absorb_short_regions(
            [
                LangRegion("en", 0.0, 40.0),
                LangRegion("fr", 40.0, 43.0),
                LangRegion("es", 43.0, 46.0),
                LangRegion("en", 46.0, 90.0),
            ],
            min_region_s=10.0,
        )
    ) == [("en", 0.0, 90.0)]


# ----- detect_language_regions_fast (single-pass fast path) -----------------
#
# The whisper-backed detection is monkeypatched, so these exercise the routing
# logic — confident → one region, uncertain / un-routable → posterior scan,
# empty → nothing — without loading a model.


def test_fast_path_routing_and_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast LID short-circuits only when confident and routable; else it defers.

    Walks every fast-path branch through one flow:
    (a) confident + routable → a single whole-file region, and the slow scan
        must NOT run; (b) low confidence → fall back to the posterior scan;
    (c) confident but un-routable language → also fall back;
    (d) empty audio → no region invented (discovery, not defaulting).
    """
    pcm5 = np.zeros(16_000 * 5, dtype=np.float32)

    # (a) Confident + routable short-circuits to one region; the fallback scan
    # is booby-trapped so any call fails the test.
    monkeypatch.setattr(lid, "detect_language", lambda pcm, **kw: ("en", 0.99))
    monkeypatch.setattr(
        lid,
        "detect_language_regions",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )
    out = detect_language_regions_fast(pcm5, 16_000)
    assert _spans(out) == [("en", 0.0, 5.0)]

    # (b) Below the confidence gate → the robust multi-window scan runs instead.
    sentinel_lowconf = [LangRegion("fr", 0.0, 3.0), LangRegion("es", 3.0, 6.0)]
    monkeypatch.setattr(lid, "detect_language", lambda pcm, **kw: ("en", 0.2))
    monkeypatch.setattr(lid, "detect_language_regions", lambda *a, **k: sentinel_lowconf)
    pcm6 = np.zeros(16_000 * 6, dtype=np.float32)
    assert (
        detect_language_regions_fast(pcm6, 16_000, conf_gate=DEFAULT_FAST_CONF_GATE)
        is sentinel_lowconf
    )

    # (c) Confident but un-routable ("zz" ∉ supported) → still defer to the scan.
    sentinel_unsup = [LangRegion("en", 0.0, 4.0)]
    monkeypatch.setattr(lid, "detect_language", lambda pcm, **kw: ("zz", 0.99))
    monkeypatch.setattr(lid, "detect_language_regions", lambda *a, **k: sentinel_unsup)
    pcm4 = np.zeros(16_000 * 4, dtype=np.float32)
    assert detect_language_regions_fast(pcm4, 16_000, supported=("en", "fr")) is sentinel_unsup

    # (d) Empty audio has no language to discover → no region is invented.
    assert detect_language_regions_fast(np.zeros(0, dtype=np.float32), 16_000) == []


# ----- detect_language discovery vs opt-in routing (no whisper.cpp) ----------


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


def test_detect_language_discovery_then_opt_in_whitelist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detection discovers whisper's true argmax by default, but re-ranks strictly
    within a caller-supplied ``supported`` set when one is given.

    Pins the discovery-first policy against one whisper posterior whose argmax
    (Galician ``gl``) is *not* in the routing hint:
    - no whitelist → the true argmax survives verbatim (never coerced away);
    - ``supported=("en","es")`` → the guard picks the best *routable* code (es),
      dropping the un-routable argmax.
    """
    dist = {"gl": 0.82, "es": 0.1, "en": 0.05}
    fake = _FakeStage(_FakeModel("gl", 0.82, dist))
    monkeypatch.setattr(lid, "_get_stage", lambda model, threads: fake)
    pcm = np.zeros(16_000, dtype=np.float32)

    # Pure discovery: the un-routable Galician argmax is surfaced as-is.
    code, prob = lid.detect_language(pcm)
    assert code == "gl"
    assert prob == pytest.approx(0.82)

    # Opt-in restriction: re-rank strictly within the routable set.
    code, prob = lid.detect_language(pcm, supported=("en", "es"))
    assert code == "es"  # best code that the caller can actually route
    assert prob == pytest.approx(0.1)
