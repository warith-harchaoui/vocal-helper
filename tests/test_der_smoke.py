"""
DER-regression smoke test — CI-safe, no model weights, no downloads.

Module summary
--------------
The full offline DER/WER regression (``test_offline_regression.py``) needs the
pyannote weights, the whisper model, and a hosted AMI subset — far too heavy to
run on every push (GitHub would throttle it). This module is the **light subset**
that *does* run in CI: it guards the two things that can silently regress without
any model at all —

1. the router's **published quality (DER) and speed (RTF) numbers**
   (``vocal_helper.router._PROFILE``), which encode the study verdict the whole
   toolbox is sold on. If a refactor quietly edits one, this fails.
2. the DER *measurement harness* itself, computed on tiny synthetic speaker
   turns via ``pyannote.metrics`` — skipped cleanly when that (light) dep is
   absent, so CI never breaks on it.

The heavy, model-backed end-to-end regression stays behind the ``integration``
marker for manual / nightly runs.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import pytest

from vocal_helper.router import _PROFILE, select_diarization

# The exact numbers the study validated on-machine (2026-07-19). Pinning them
# here turns any silent edit of the scientific claims into a red test — the DER
# regression that needs no models to catch.
EXPECTED_PROFILE: dict[tuple[str, str], tuple[float, float]] = {
    ("offline", "nemo"): (0.142, 0.051),
    ("offline", "pyannote"): (0.122, 0.067),
    ("offline", "sherpa"): (0.174, 0.58),
    ("online", "nemo"): (0.586, 0.030),
    ("online", "sherpa"): (0.174, 0.58),
}


def test_router_profile_numbers_are_pinned() -> None:
    """Every published (DER, RTF) matches the validated study value exactly."""
    # A drift guard: the router's numbers are its scientific promise, so an
    # accidental edit (or a "tidy-up" that rounds them) must fail loudly.
    assert _PROFILE == EXPECTED_PROFILE


def test_router_der_rtf_are_physically_plausible() -> None:
    """Every DER sits in [0, 1] and every RTF is positive — sanity envelope."""
    # DER is an error *rate* (0 = perfect, 1 = every frame wrong); RTF is a
    # positive time ratio. Anything outside these bounds is a data-entry bug.
    for (mode, backend), (der, rtf) in _PROFILE.items():
        assert 0.0 <= der <= 1.0, f"{mode}/{backend} DER {der} out of [0,1]"
        assert rtf > 0.0, f"{mode}/{backend} RTF {rtf} must be positive"


def test_router_scenarios_surface_quality_and_speed() -> None:
    """Each routed scenario reports the DER + RTF that back its decision."""
    # Short offline → nemo, with its study numbers attached (quality + speed are
    # first-class, not a bare backend name).
    short = select_diarization(live=False, duration_s=45.0, max_speakers=2)
    assert short.backend == "nemo"
    assert (short.expected_der, short.expected_rtf) == _PROFILE[("offline", "nemo")]

    # Long offline → pyannote, the robust default past the NeMo ceiling.
    long = select_diarization(live=False, duration_s=1800.0)
    assert long.backend == "pyannote"
    assert (long.expected_der, long.expected_rtf) == _PROFILE[("offline", "pyannote")]

    # Live stream → nemo, the best online embedder at every length.
    live = select_diarization(live=True)
    assert live.backend == "nemo"
    assert (live.expected_der, live.expected_rtf) == _PROFILE[("online", "nemo")]


def test_der_metric_on_synthetic_turns() -> None:
    """DER computes correctly on hand-built turns (harness guard, no models)."""
    # pyannote.metrics is light (no torch) but optional in the fast venv — skip
    # cleanly rather than fail CI when it is not installed.
    pytest.importorskip("pyannote.core")
    pytest.importorskip("pyannote.metrics")
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    # Reference: two clean, non-overlapping turns by two speakers.
    ref = Annotation()
    ref[Segment(0.0, 2.0)] = "A"
    ref[Segment(2.0, 4.0)] = "B"

    # An identical hypothesis must score DER 0 — the metric's fixed point.
    perfect = Annotation()
    perfect[Segment(0.0, 2.0)] = "A"
    perfect[Segment(2.0, 4.0)] = "B"
    metric = DiarizationErrorRate(collar=0.0)
    assert metric(ref, perfect) == pytest.approx(0.0, abs=1e-9)

    # A hypothesis that misses the entire second speaker turn is 50 % missed
    # detection over 4 s of reference speech → DER 0.5, a known anchor value.
    half_missed = Annotation()
    half_missed[Segment(0.0, 2.0)] = "A"
    assert DiarizationErrorRate(collar=0.0)(ref, half_missed) == pytest.approx(0.5, abs=1e-6)
