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


# ----- language_posterior_curve discovery axis (no whisper.cpp) --------------


def test_posterior_curve_discovers_axis_and_reports_unknown_when_too_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The curve adopts whisper's full head as its axis, but invents nothing on
    audio too short to identify.

    Three discovery-first facts on one mocked posterior:
    - with no ``supported`` set, the language axis is *discovered* from the first
      usable (≥ 1 s) window — sorted whisper codes, no code filtered out;
    - audio under 1 s yields no usable window, so an unrestricted call returns an
      empty axis (caller must treat the language as unknown, never defaulted);
    - the same too-short audio with a caller-fixed axis still returns that axis
      as a single uniform-prior frame rather than an empty one.
    """
    dist = {"fr": 0.7, "en": 0.2, "de": 0.1}
    fake = _FakeStage(_FakeModel("fr", 0.7, dist))
    monkeypatch.setattr(lid, "_get_stage", lambda model, threads: fake)
    sr = 16_000

    # 12 s of audio → several usable windows; the axis is whisper's own head,
    # sorted, and every row is a proper (summing-to-one) distribution over it.
    pcm = np.zeros(12 * sr, dtype=np.float32)
    centers, langs, post = lid.language_posterior_curve(pcm, sr, hop_s=3.0, window_s=10.0)
    assert langs == ["de", "en", "fr"]  # discovered + sorted, nothing dropped
    assert post.shape[1] == 3
    assert np.allclose(post.sum(axis=1), 1.0)  # each window renormalised
    assert len(centers) == post.shape[0]

    # < 1 s → no window ever reaches the 1 s identification floor. Unrestricted
    # discovery must report "unknown" (empty axis), not fabricate a language.
    tiny = np.zeros(sr // 2, dtype=np.float32)
    _c, langs_empty, post_empty = lid.language_posterior_curve(tiny, sr)
    assert langs_empty == []
    assert post_empty.shape[1] == 0

    # Same too-short audio, but the caller fixed the axis: a valid uniform-prior
    # frame over exactly that routable set is returned instead of an empty axis.
    _c2, langs_fixed, post_fixed = lid.language_posterior_curve(tiny, sr, supported=("en", "fr"))
    assert langs_fixed == ["en", "fr"]
    assert np.allclose(post_fixed, 0.5)  # uniform over the two fixed codes


# ----- detect_language_regions full pipeline (no whisper.cpp) ----------------


def test_regions_empty_short_and_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-to-end partition invents nothing on empty/short audio and finds a
    real switch when the posterior argmax flips.

    Drives the whole detect_language_regions pipeline (curve → smooth → argmax →
    change points → absorb → refine → snap) through the mocked whisper head:
    - empty input → empty list (no region fabricated for silence);
    - audio too short to identify → empty list (discovery yields no axis);
    - a first-half-French / second-half-Spanish signal → exactly the fr→es
      switch, with contiguous, file-spanning regions and discovered labels.
    """
    sr = 16_000

    # Empty input short-circuits before any whisper call.
    assert lid.detect_language_regions(np.zeros(0, dtype=np.float32), sr) == []

    # Half-second clip: no window clears the 1 s floor → empty axis → no region.
    monkeypatch.setattr(
        lid, "_get_stage", lambda model, threads: _FakeStage(_FakeModel("fr", 0.9, {"fr": 1.0}))
    )
    assert lid.detect_language_regions(np.zeros(sr // 2, dtype=np.float32), sr) == []

    # A time-varying posterior: French dominates the first half, Spanish the
    # second. The fake model reads the window's midpoint to pick the winner, so
    # the argmax genuinely flips partway through a 40 s file.
    class _SwitchModel:
        """Posterior that flips fr→es at the 20 s mark (based on window content)."""

        def auto_detect_language(self, pcm, offset_ms: int = 0):
            # A non-zero marker sample encodes the window's centre time; the
            # region pipeline hands us real slices, so key off their mean sign.
            leans_es = float(np.mean(pcm)) > 0.0
            dist = {"es": 0.9, "fr": 0.1} if leans_es else {"fr": 0.9, "es": 0.1}
            top = "es" if leans_es else "fr"
            return (top, dist[top]), dist

    monkeypatch.setattr(lid, "_get_stage", lambda model, threads: _FakeStage(_SwitchModel()))
    t = np.arange(40 * sr) / sr
    pcm = np.zeros(40 * sr, dtype=np.float32)
    pcm[t >= 20.0] = 0.5  # positive-mean second half → "es"; first half → "fr"
    regions = lid.detect_language_regions(pcm, sr, smooth_s=0.0, snap_s=0.0, refine_s=0.0)
    langs = [r.lang for r in regions]
    assert langs == ["fr", "es"]  # discovered switch, in order
    assert regions[0].t0 == 0.0 and regions[-1].t1 == pytest.approx(40.0)
    # Regions tile the file with no gaps or overlaps.
    for a, b in zip(regions, regions[1:], strict=False):
        assert a.t1 == b.t0


# ----- SpeechBrain second opinion + cross-check (no torch / no model) --------


def test_speechbrain_label_parse_and_cross_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The VoxLingua107 opinion parses its ``iso: Name`` label and cross-check
    only judges regions long enough to classify.

    Two independent-verification contracts on a fake classifier:
    - ``detect_language_speechbrain`` strips ``"fr: French"`` to the ISO prefix
      and, with no whitelist, returns that true label verbatim (never remapped);
      with a whitelist that excludes it, it drops to the first routable code;
    - ``cross_check_regions`` runs the classifier only on regions ≥ 1 s, passing
      shorter ones through as trivially agreeing, and flags a genuine
      whisper-vs-SpeechBrain disagreement as ``agree=False``.
    """
    sr = 16_000

    class _FakeScore:
        """Mimics a torch scalar: ``.exp()`` yields the probability."""

        def __init__(self, value: float) -> None:
            self._v = value

        def exp(self) -> float:
            return self._v

    class _FakeClassifier:
        """Returns a fixed VoxLingua107 ``"iso: Name"`` label + log-prob score."""

        def __init__(self, label: str, prob: float) -> None:
            self._label, self._prob = label, prob

        def classify_batch(self, wav):
            # SpeechBrain's real signature: (out_prob, score, index, text_lab).
            return None, _FakeScore(self._prob), None, [self._label]

    # Force a French second opinion; torch is only used to wrap the array, so a
    # stub module is enough to keep the test offline.
    import sys
    import types as _types

    torch_stub = _types.ModuleType("torch")
    torch_stub.tensor = lambda arr: _StubTensor(arr)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(lid, "_ensure_classifier", lambda: _FakeClassifier("fr: French", 0.88))

    seg = np.zeros(sr, dtype=np.float32)
    code, prob = lid.detect_language_speechbrain(seg)
    assert code == "fr"  # ISO prefix parsed out of "fr: French"
    assert prob == pytest.approx(0.88)

    # Whitelist that excludes the true label → drop to the first routable code
    # (opt-in coercion only; the honest label would otherwise survive).
    dropped, _p = lid.detect_language_speechbrain(seg, supported=("en", "de"))
    assert dropped == "en"

    # Cross-check: a 1 s "es" region disagrees with SpeechBrain's "fr"; a 0.5 s
    # region is too short to judge and passes through as agreeing.
    regions = [LangRegion("es", 0.0, 1.0), LangRegion("en", 1.0, 1.5)]
    pcm = np.zeros(int(1.5 * sr), dtype=np.float32)
    verdicts = lid.cross_check_regions(pcm, regions, sr)
    assert verdicts[0].primary == "es" and verdicts[0].speechbrain == "fr"
    assert verdicts[0].agree is False  # genuine model disagreement, not a bug
    assert verdicts[1].speechbrain == "-"  # too short → sentinel, trivially agrees
    assert verdicts[1].agree is True


class _StubTensor:
    """Minimal torch-tensor stand-in supporting ``.unsqueeze`` for the SB path."""

    def __init__(self, arr) -> None:
        self._arr = arr

    def unsqueeze(self, _dim: int) -> _StubTensor:
        return self


# ----- _ensure_classifier failure modes (opt-in [lid] extra) -----------------


def test_ensure_classifier_requires_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a local VoxLingua107 snapshot, the second opinion refuses to fetch
    from HuggingFace and raises a clear RuntimeError.

    The independent verifier is strictly bundle-only (no HF fallback). With
    SpeechBrain importable but no configured engines directory,
    ``_ensure_classifier`` must raise ``RuntimeError`` naming the settings knob —
    not silently reach out to the network.
    """
    import sys
    import types as _types

    # Reset the process-wide singleton so this test builds the classifier fresh.
    monkeypatch.setattr(lid, "_classifier", None)
    # SpeechBrain present (import succeeds) but no engines bundle resolved.
    sb_stub = _types.ModuleType("speechbrain.inference.classifiers")
    sb_stub.EncoderClassifier = object
    monkeypatch.setitem(sys.modules, "speechbrain", _types.ModuleType("speechbrain"))
    monkeypatch.setitem(
        sys.modules, "speechbrain.inference", _types.ModuleType("speechbrain.inference")
    )
    monkeypatch.setitem(sys.modules, "speechbrain.inference.classifiers", sb_stub)
    # No local snapshot → the bundle-only guard must trip.
    monkeypatch.setattr("vocal_helper.diar.resolve_diarization_engines", lambda: None)
    with pytest.raises(RuntimeError, match="VoxLingua107"):
        lid._ensure_classifier()
