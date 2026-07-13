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

from vocal_helper.lid import (
    DEFAULT_MIN_REGION_S,
    DEFAULT_SUPPORTED_LANGS,
    DEFAULT_WINDOW_S,
    LangRegion,
    _absorb_short_regions,
    _coalesce,
    _snap_boundaries_to_silence,
)


def _spans(regions: list[LangRegion]) -> list[tuple[str, float, float]]:
    return [(r.lang, r.t0, r.t1) for r in regions]


# ----- defaults -------------------------------------------------------------


def test_lid_defaults() -> None:
    assert DEFAULT_WINDOW_S == 10.0
    assert DEFAULT_MIN_REGION_S == 8.0
    # Broad ISO-639-1 set, en first (the empty-input fallback language).
    assert DEFAULT_SUPPORTED_LANGS[0] == "en"
    assert {"en", "fr", "es", "it", "pl", "nl"} <= set(DEFAULT_SUPPORTED_LANGS)
    # Canonical ISO-639-1 : two-letter, lower-case, no region suffixes.
    assert all(len(c) == 2 and c.islower() for c in DEFAULT_SUPPORTED_LANGS)


def test_snap_boundary_moves_to_silence() -> None:
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
    assert _coalesce([]) == []
    one = [LangRegion("fr", 0.0, 30.0)]
    assert _spans(_coalesce(one)) == [("fr", 0.0, 30.0)]


def test_coalesce_no_adjacent_duplicates_is_identity() -> None:
    regions = [
        LangRegion("en", 0.0, 20.0),
        LangRegion("fr", 20.0, 40.0),
        LangRegion("en", 40.0, 60.0),
    ]
    assert _spans(_coalesce(regions)) == _spans(regions)


# ----- _absorb_short_regions ------------------------------------------------


def test_absorb_short_region_into_longer_neighbour() -> None:
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
    # Both regions clear the threshold → a genuine switch, left untouched.
    regions = [
        LangRegion("en", 0.0, 30.0),
        LangRegion("fr", 30.0, 60.0),
    ]
    absorbed = _absorb_short_regions(regions, min_region_s=DEFAULT_MIN_REGION_S)
    assert _spans(absorbed) == _spans(regions)


def test_absorb_short_region_prefers_longer_side() -> None:
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
    # Leading short region has no left neighbour → absorbed rightward.
    regions = [
        LangRegion("fr", 0.0, 3.0),
        LangRegion("en", 3.0, 40.0),
    ]
    absorbed = _absorb_short_regions(regions, min_region_s=10.0)
    assert _spans(absorbed) == [("en", 0.0, 40.0)]


def test_absorb_single_region_untouched_even_if_short() -> None:
    # A lone region is always returned — a short monolingual file must not
    # collapse to nothing.
    one = [LangRegion("nl", 0.0, 2.0)]
    assert _spans(_absorb_short_regions(one, min_region_s=10.0)) == [("nl", 0.0, 2.0)]


def test_absorb_cascades_until_all_clear() -> None:
    # Several short blips collapse into the dominant surrounding language.
    regions = [
        LangRegion("en", 0.0, 40.0),
        LangRegion("fr", 40.0, 43.0),
        LangRegion("es", 43.0, 46.0),
        LangRegion("en", 46.0, 90.0),
    ]
    absorbed = _absorb_short_regions(regions, min_region_s=10.0)
    assert _spans(absorbed) == [("en", 0.0, 90.0)]
