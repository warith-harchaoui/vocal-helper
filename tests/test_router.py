"""Backend router (the aiguilleur) — the study-grounded diarizer selection.

Pure decision logic, no models: every rule from :func:`select_diarization` is
exercised over the scenarios that move DER (quality) and RTF (speed) —
live-vs-batch, duration, speaker count, torch-availability, pyannote-
availability — AND the quality/speed numbers each decision reports are pinned.
This is where "which backend, when, at what quality and speed" is proven.

The micro-tests were consolidated into a handful of table-driven scenario
tests (CODING.md §15): one offline-crossover table, one streaming table, one
torch-free / fallback table, plus the profile-invariant and dataclass guards —
without dropping any routed scenario the repo's headline claim rests on.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
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


@pytest.mark.parametrize(
    ("duration_s", "max_speakers", "backend", "der", "rtf", "reason_needle"),
    [
        # Short, ≤4-speaker batch → NeMo Sortformer, the crossover's fast branch;
        # DER/RTF are first-class and match the on-machine run.
        (45.0, 3, "nemo", 0.142, 0.051, "nemo"),
        # Long-form batch → pyannote (nemo hangs past ~25 min); best AMI quality.
        (1800.0, None, "pyannote", 0.122, 0.067, "pyannote"),
        # Exactly at the ceiling still counts as short (≤ is inclusive) → nemo.
        (NEMO_MAX_DURATION_S, 2, "nemo", 0.142, 0.051, "nemo"),
        # One second past the ceiling flips to the long-form backend.
        (NEMO_MAX_DURATION_S + 1.0, None, "pyannote", 0.122, 0.067, "pyannote"),
        # Unknown length must NOT be optimistically routed to the capped backend.
        (None, None, "pyannote", 0.122, 0.067, "pyannote"),
        # Short but >4 known speakers skips Sortformer's 4-speaker cap → pyannote.
        (30.0, SORTFORMER_MAX_SPEAKERS + 1, "pyannote", 0.122, 0.067, "pyannote"),
    ],
)
def test_offline_routing_by_duration_and_speakers(
    duration_s: float | None,
    max_speakers: int | None,
    backend: str,
    der: float,
    rtf: float,
    reason_needle: str,
) -> None:
    """Offline crossover table: duration + speaker cap pick the backend, DER and RTF.

    Sweeps the quality/speed crossover the aiguilleur is built on — the short →
    nemo / long → pyannote boundary (inclusive ceiling, the +1 s flip), the
    unknown-duration long-form default, and the 4-speaker Sortformer cap — and
    pins the exact backend, expected DER and expected RTF each decision reports.

    Parameters
    ----------
    duration_s : float or None
        Probed clip length; ``None`` models a failed duration probe.
    max_speakers : int or None
        Known speaker count, or ``None`` when unspecified.
    backend : str
        Backend the router must select for this scenario.
    der, rtf : float
        Quality (DER) and speed (RTF) the plan must report, from ``_PROFILE``.
    reason_needle : str
        Substring the human-readable rationale must contain.
    """
    plan = select_diarization(live=False, duration_s=duration_s, max_speakers=max_speakers)
    assert plan.mode == "offline"
    assert plan.backend == backend
    assert (plan.expected_der, plan.expected_rtf) == (der, rtf)
    assert reason_needle in plan.reason.lower()


# ----- streaming (online) scenarios -----------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_s": 30.0, "max_speakers": 2},  # short live stream
        {"duration_s": None},  # unknown-length live stream
        {"duration_s": 9000.0},  # very long live stream
        {"duration_s": 30.0, "max_speakers": 8},  # crowded stream (>4 speakers)
    ],
)
def test_stream_always_picks_online_nemo(kwargs: dict) -> None:
    """Live audio → online nemo at every length and speaker count.

    vocal-helper's online nemo beats online pyannote at both lengths on this
    machine, so streaming never routes to pyannote — short, long and unknown all
    resolve to nemo. The 4-speaker Sortformer cap is offline-only, so a crowded
    live stream (TitaNet-clustered) still routes to online nemo, not pyannote.

    Parameters
    ----------
    kwargs : dict
        Streaming scenario passed through to :func:`select_diarization` with
        ``live=True``.
    """
    plan = select_diarization(live=True, **kwargs)  # type: ignore[arg-type]
    assert (plan.mode, plan.backend) == ("online", "nemo")
    # The streaming numbers differ from offline — latency-bound approximation.
    assert (plan.expected_der, plan.expected_rtf) == (0.586, 0.030)


# ----- torch-free / fallback scenarios --------------------------------------


def test_torch_free_forces_sherpa_and_notes_rediarization() -> None:
    """A torch-free deployment always gets sherpa; live sherpa is periodic re-diarization.

    Covers the onnxruntime backend selection regardless of length or live-ness
    (short offline, very-long online) and the ADR-0002 rationale that torch-free
    live sherpa is periodic offline re-diarization rather than true streaming.
    """
    # Short offline and very-long online both resolve to sherpa without torch.
    assert select_diarization(live=False, duration_s=20.0, torch_free=True).backend == "sherpa"
    assert select_diarization(live=True, duration_s=9999.0, torch_free=True).backend == "sherpa"
    # The live torch-free rationale spells out the periodic-re-diarization trick.
    plan = select_diarization(live=True, torch_free=True)
    assert plan.backend == "sherpa"
    assert "periodic offline re-diarization" in plan.reason


def test_pyannote_unavailable_falls_back_to_sherpa_not_nemo() -> None:
    """On the long-form branch, missing pyannote falls back to sherpa (nemo is unsafe here)."""
    plan = select_diarization(live=False, duration_s=3600.0, pyannote_available=False)
    assert plan.backend == "sherpa"


@pytest.mark.parametrize("live,expected_mode", [(True, "online"), (False, "offline")])
def test_mode_tracks_live_flag(live: bool, expected_mode: str) -> None:
    """``live`` selects the streaming vs whole-buffer mode independently of backend.

    Parameters
    ----------
    live : bool
        Whether the request is a live stream.
    expected_mode : str
        Mode the router must report for that ``live`` flag.
    """
    plan = select_diarization(live=live, duration_s=45.0, max_speakers=2)
    assert plan.mode == expected_mode


# ----- invariants on the reported quality/speed -----------------------------


def test_every_plan_reports_self_consistent_profile_numbers() -> None:
    """Every emitted plan sources its DER/RTF from ``_PROFILE``, and the table's trade-offs hold.

    First sweeps the meaningful scenario corners and asserts the DER/RTF fields
    are read from the single quality/speed table (never hand-typed at a call
    site). Then pins the headline trade-offs the router relies on: offline nemo
    is the faster backend, online is a ~3-4x-worse-DER latency approximation,
    and sherpa trades a little quality for portability.
    """
    # Sweep the corners and confirm each plan's numbers come from the table.
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

    # Offline: nemo Sortformer is the *faster* backend (lower RTF than pyannote).
    assert _PROFILE[("offline", "nemo")][1] < _PROFILE[("offline", "pyannote")][1]
    # Online is a latency-bound approximation — much worse quality than offline.
    assert _PROFILE[("online", "nemo")][0] > 3 * _PROFILE[("offline", "nemo")][0]
    # sherpa trades a little quality for portability — worse DER than pyannote.
    assert _PROFILE[("offline", "sherpa")][0] > _PROFILE[("offline", "pyannote")][0]


def test_returns_backend_plan_dataclass() -> None:
    """The router returns a frozen :class:`BackendPlan` with a justified, positive-metric plan."""
    plan = select_diarization(live=True)
    assert isinstance(plan, BackendPlan)
    assert plan.reason  # every decision is justified
    assert plan.expected_der > 0 and plan.expected_rtf > 0


def test_known_count_pins_sherpa_clustering() -> None:
    """A known speaker count pins sherpa's clustering; absence leaves it on auto.

    This is the pdbms diar-study §12.1 fix: a torch-free 2-party call must carry the
    count into the plan so the consumer collapses sherpa's over-segmentation.
    """
    pinned = select_diarization(live=False, torch_free=True, num_speakers=2)
    assert pinned.backend == "sherpa"
    assert pinned.sherpa_num_clusters == 2

    # No known count → auto-clustering, so meeting-audio behaviour is unchanged.
    auto = select_diarization(live=False, torch_free=True)
    assert auto.backend == "sherpa"
    assert auto.sherpa_num_clusters is None

    # The count also rides the pyannote-unavailable → sherpa fallback.
    fallback = select_diarization(live=False, duration_s=5000.0,
                                  pyannote_available=False, num_speakers=2)
    assert fallback.backend == "sherpa"
    assert fallback.sherpa_num_clusters == 2

    # A known count never perturbs a non-sherpa pick (no pin field on nemo/pyannote).
    nemo = select_diarization(live=False, duration_s=45.0, num_speakers=2)
    assert nemo.backend == "nemo"
    assert nemo.sherpa_num_clusters is None


def test_rationale_aliases_reason() -> None:
    """``plan.rationale`` mirrors ``plan.reason`` (consumers log the former)."""
    plan = select_diarization(live=True)
    assert plan.rationale == plan.reason
