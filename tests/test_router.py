"""Backend router (the aiguilleur) — the study-grounded diarizer selection.

Pure decision logic, no models: every rule from :func:`select_diarization` is
exercised over the scenarios that move DER (quality) and RTF (speed) —
live-vs-batch, duration, speaker count, torch-availability, pyannote-
availability — AND the quality/speed numbers each decision reports are pinned.
This is where "which backend, when, at what quality and speed" is proven.
"""

from __future__ import annotations

import pytest

from vocal_helper.router import (
    _PROFILE,
    NEMO_MAX_DURATION_S,
    SORTFORMER_MAX_SPEAKERS,
    BackendPlan,
    select_diarization,
)

# ----- offline scenarios ----------------------------------------------------


def test_short_batch_low_speakers_picks_nemo() -> None:
    """Short, ≤4-speaker batch audio → NeMo Sortformer, with its quality+speed."""
    plan = select_diarization(live=False, duration_s=45.0, max_speakers=3)
    assert (plan.mode, plan.backend) == ("offline", "nemo")
    # Quality (DER) and speed (RTF) are first-class and match the on-machine run.
    assert plan.expected_der == 0.142
    assert plan.expected_rtf == 0.051
    assert "nemo" in plan.reason.lower()


def test_long_batch_picks_pyannote() -> None:
    """Long-form batch audio → pyannote (nemo hangs past ~25 min)."""
    plan = select_diarization(live=False, duration_s=1800.0)
    assert (plan.mode, plan.backend) == ("offline", "pyannote")
    assert plan.expected_der == 0.122  # AMI dev-slice median (best quality)
    assert plan.expected_rtf == 0.067


def test_unknown_duration_is_treated_as_long_form() -> None:
    """No duration → the robust pyannote branch, never a guessed 'short'."""
    # A missing length must not be optimistically routed to the capped backend.
    plan = select_diarization(live=False, duration_s=None)
    assert plan.backend == "pyannote"


def test_short_but_too_many_speakers_avoids_nemo() -> None:
    """A short clip with >4 known speakers skips Sortformer's 4-speaker cap."""
    plan = select_diarization(live=False, duration_s=30.0, max_speakers=SORTFORMER_MAX_SPEAKERS + 1)
    assert plan.backend == "pyannote"


def test_boundary_duration_is_inclusive_for_nemo() -> None:
    """Exactly at the ceiling still counts as short (≤ is inclusive)."""
    plan = select_diarization(live=False, duration_s=NEMO_MAX_DURATION_S, max_speakers=2)
    assert plan.backend == "nemo"


def test_just_over_ceiling_flips_to_pyannote() -> None:
    """One second past the ceiling flips to the long-form backend."""
    plan = select_diarization(live=False, duration_s=NEMO_MAX_DURATION_S + 1.0)
    assert plan.backend == "pyannote"


# ----- streaming (online) scenarios -----------------------------------------


def test_stream_always_picks_online_nemo() -> None:
    """Live audio → online nemo at *every* length (no online length crossover).

    vocal-helper's OnlineDiarStage online nemo beats online pyannote at both
    lengths on this machine (0.586/0.497 vs 0.590/0.844), so streaming never
    routes to pyannote — short, long and unknown all resolve to nemo.
    """
    for kw in (
        {"duration_s": 30.0, "max_speakers": 2},
        {"duration_s": None},
        {"duration_s": 9000.0},
    ):
        plan = select_diarization(live=True, **kw)  # type: ignore[arg-type]
        assert (plan.mode, plan.backend) == ("online", "nemo")
    # The streaming numbers differ from offline — latency-bound approximation.
    plan = select_diarization(live=True, duration_s=None)
    assert plan.expected_der == 0.586
    assert plan.expected_rtf == 0.030


def test_stream_ignores_speaker_cap() -> None:
    """The 4-speaker Sortformer cap is offline-only; online nemo (TitaNet) is uncapped."""
    # A crowded live stream still routes to online nemo, not pyannote.
    plan = select_diarization(live=True, duration_s=30.0, max_speakers=8)
    assert (plan.mode, plan.backend) == ("online", "nemo")


# ----- torch-free / fallback scenarios --------------------------------------


def test_torch_free_forces_sherpa_regardless_of_length() -> None:
    """A torch-free deployment always gets the onnxruntime backend."""
    # Short and long both resolve to sherpa when PyTorch is unavailable.
    assert select_diarization(live=False, duration_s=20.0, torch_free=True).backend == "sherpa"
    assert select_diarization(live=True, duration_s=9999.0, torch_free=True).backend == "sherpa"


def test_torch_free_streaming_notes_periodic_rediarization() -> None:
    """torch-free live sherpa is periodic offline re-diarization (ADR 0002)."""
    plan = select_diarization(live=True, torch_free=True)
    assert plan.backend == "sherpa"
    assert "periodic offline re-diarization" in plan.reason


def test_pyannote_unavailable_falls_back_to_sherpa_not_nemo() -> None:
    """On the long-form branch, missing pyannote falls back to sherpa (nemo is unsafe here)."""
    plan = select_diarization(live=False, duration_s=3600.0, pyannote_available=False)
    assert plan.backend == "sherpa"


@pytest.mark.parametrize("live,expected_mode", [(True, "online"), (False, "offline")])
def test_mode_tracks_live_flag(live: bool, expected_mode: str) -> None:
    """``live`` selects the streaming vs whole-buffer mode independently of backend."""
    plan = select_diarization(live=live, duration_s=45.0, max_speakers=2)
    assert plan.mode == expected_mode


# ----- invariants on the reported quality/speed -----------------------------


def test_every_plan_reports_profile_numbers() -> None:
    """Whatever the router emits, its DER/RTF match the ``_PROFILE`` table exactly."""
    # Sweep the meaningful scenario corners and assert the fields are sourced
    # from the single quality/speed table (never hand-typed at a call site).
    scenarios = [
        {"live": False, "duration_s": 30.0, "max_speakers": 2},
        {"live": False, "duration_s": 5000.0},
        {"live": True, "duration_s": 30.0, "max_speakers": 2},
        {"live": True, "duration_s": None},
        {"live": False, "torch_free": True},
        {"live": True, "torch_free": True},
        {"live": False, "duration_s": 5000.0, "pyannote_available": False},
    ]
    for kw in scenarios:
        plan = select_diarization(**kw)  # type: ignore[arg-type]
        assert (plan.expected_der, plan.expected_rtf) == _PROFILE[(plan.mode, plan.backend)]


def test_profile_is_self_consistent_on_quality_and_speed() -> None:
    """The evidence table encodes the headline trade-offs the router relies on."""
    # Offline: nemo Sortformer is the *faster* backend (lower RTF than pyannote).
    assert _PROFILE[("offline", "nemo")][1] < _PROFILE[("offline", "pyannote")][1]
    # Online is a latency-bound approximation — much worse quality than offline
    # (the ~3-4x-offline DER the online branch's reason warns about).
    assert _PROFILE[("online", "nemo")][0] > 3 * _PROFILE[("offline", "nemo")][0]
    # sherpa trades a little quality for portability — worse DER than pyannote.
    assert _PROFILE[("offline", "sherpa")][0] > _PROFILE[("offline", "pyannote")][0]


def test_returns_backend_plan_dataclass() -> None:
    """The router returns a frozen :class:`BackendPlan` with a non-empty reason."""
    plan = select_diarization(live=True)
    assert isinstance(plan, BackendPlan)
    assert plan.reason  # every decision is justified
    assert plan.expected_der > 0 and plan.expected_rtf > 0
