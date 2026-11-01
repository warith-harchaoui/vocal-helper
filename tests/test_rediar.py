"""Offline-quality diarization on a live stream, at a cost that does not grow.

The priority this module answers is explicit: the labels matter as much as the
words, and the only hard constraint is that the whole system stays under a
real-time factor of 1. So the tests check two things — that the quality path is
actually reachable within that budget, and that the identity strategy ported from
the native block-wise diarizer keeps its three load-bearing details.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import numpy as np
import pytest

from vocal_helper.rediar import (
    DEFAULT_GAP_S,
    RediarPolicy,
    SpeakerReference,
    plan_refresh,
    read_back_identities,
)

# Measured on this suite's own material (studies/asr_acceleration_rtf.md and router).
ASR_METAL, ASR_CPU = 0.033, 0.164
OFFLINE_PYANNOTE, OFFLINE_SHERPA = 0.067, 0.580


# --------------------------------------------------------------------------- #
#  The cost: bounded in time, mild in speakers
# --------------------------------------------------------------------------- #


def test_the_cost_does_not_grow_with_the_session():
    """The whole reason the window is bounded. A policy whose cost depended on how
    long the conversation had been running would work in a demo and fail in use."""
    policy = RediarPolicy()
    first = policy.stage_rtf(OFFLINE_PYANNOTE, speakers=4)
    # Same policy, same speakers — the session length is simply not an input.
    assert policy.stage_rtf(OFFLINE_PYANNOTE, speakers=4) == first
    assert "buffer" not in RediarPolicy.stage_rtf.__code__.co_varnames


def test_the_unbounded_version_is_kept_measurable_because_the_number_is_the_argument():
    """Three minutes looks fine, one hour is impossible. Keeping it computable is
    what stops "re-diarize everything" from surviving another review."""
    short = RediarPolicy.unbounded_rtf(buffer_s=180.0, period_s=60.0, backend_rtf=OFFLINE_PYANNOTE)
    hour = RediarPolicy.unbounded_rtf(buffer_s=3600.0, period_s=60.0, backend_rtf=OFFLINE_PYANNOTE)
    assert short < 0.25  # plausible
    assert hour > 3.0  # hopeless
    assert hour / short == pytest.approx(20.0)


def test_the_cost_grows_with_the_speaker_count_but_only_mildly():
    """The primer is per-speaker, so it must grow — a lump primer would miss whoever
    had not spoken yet. At the default settings each extra voice costs about 0.014,
    which is why a real panel fits comfortably."""
    policy = RediarPolicy()
    per_speaker = policy.stage_rtf(OFFLINE_PYANNOTE, speakers=2) - policy.stage_rtf(
        OFFLINE_PYANNOTE, speakers=1
    )
    assert 0.010 < per_speaker < 0.020
    assert policy.stage_rtf(OFFLINE_PYANNOTE, speakers=8) > policy.stage_rtf(
        OFFLINE_PYANNOTE, speakers=1
    )


def test_no_speakers_means_no_primer():
    """The first refresh has nothing to anchor against, so it pays nothing for it."""
    assert RediarPolicy().primer_s(speakers=0) == 0.0


def test_the_quality_path_fits_the_budget_on_a_gpu_machine():
    """The headline claim: the full offline error rate, live, with room to spare."""
    policy = RediarPolicy()
    assert policy.holds(asr_rtf=ASR_METAL, backend_rtf=OFFLINE_PYANNOTE, speakers=6)
    assert policy.max_speakers_within(asr_rtf=ASR_METAL, backend_rtf=OFFLINE_PYANNOTE) >= 12


def test_it_still_fits_a_real_panel_without_a_gpu():
    """Transcription costs five times more on the CPU, so the room shrinks — but a
    six-person panel still fits, which is the case that matters."""
    policy = RediarPolicy()
    assert policy.holds(asr_rtf=ASR_CPU, backend_rtf=OFFLINE_PYANNOTE, speakers=6)
    assert policy.max_speakers_within(asr_rtf=ASR_CPU, backend_rtf=OFFLINE_PYANNOTE) >= 6


def test_the_torch_free_diarizer_does_not_fit_and_says_so():
    """sherpa offline is 8.7 times the cost of pyannote. Re-running it periodically
    does not fit, and pretending otherwise would promise quality the machine cannot
    deliver. The honest answer is a longer period or the online stage."""
    policy = RediarPolicy(backend="sherpa")
    assert not policy.holds(asr_rtf=ASR_METAL, backend_rtf=OFFLINE_SHERPA, speakers=6)


def test_refreshing_too_often_spends_everything_on_repetition():
    """The primer is re-diarized at every refresh, so a short period pays for the
    same audio again and again. There is a floor below which more frequent
    correction buys nothing because the budget is gone."""
    assert (
        RediarPolicy(period_s=20.0).max_speakers_within(
            asr_rtf=ASR_METAL, backend_rtf=OFFLINE_PYANNOTE
        )
        == 0
    )


def test_a_wider_period_buys_speakers_at_the_price_of_staler_labels():
    """The actual trade to tune, and it should be visible in the numbers: the period
    is also the worst case for how long a wrong label stays on screen."""
    tight, relaxed = RediarPolicy(period_s=30.0), RediarPolicy(period_s=120.0)
    assert relaxed.max_speakers_within(
        asr_rtf=ASR_METAL, backend_rtf=OFFLINE_PYANNOTE
    ) > tight.max_speakers_within(asr_rtf=ASR_METAL, backend_rtf=OFFLINE_PYANNOTE)


# --------------------------------------------------------------------------- #
#  Scheduling
# --------------------------------------------------------------------------- #


def test_a_refresh_waits_for_a_full_period():
    policy = RediarPolicy(period_s=60.0)
    assert plan_refresh(now_s=59.9, last_refresh_s=0.0, policy=policy) is None
    assert plan_refresh(now_s=60.0, last_refresh_s=0.0, policy=policy) == (0.0, 60.0)


def test_an_early_refresh_asks_only_for_audio_that_exists():
    """Clamped at zero: a window wider than the session would otherwise ask for
    negative time, and whatever buffer answered that would be lying."""
    policy = RediarPolicy(window_s=180.0, period_s=60.0)
    assert plan_refresh(now_s=60.0, last_refresh_s=0.0, policy=policy) == (0.0, 60.0)


def test_a_later_refresh_slides_the_window():
    policy = RediarPolicy(window_s=180.0, period_s=60.0)
    assert plan_refresh(now_s=600.0, last_refresh_s=540.0, policy=policy) == (420.0, 600.0)


# --------------------------------------------------------------------------- #
#  Identity: read-back, not matching
# --------------------------------------------------------------------------- #


def _ref(speaker, t0, t1):
    return SpeakerReference(speaker, np.zeros(1, dtype=np.float32), t0, t1)


def test_identity_is_read_back_from_whichever_label_covered_the_reference():
    """No embedding, no centroid, no threshold of our own: the engine's own
    clustering already did the work, and this only reads its answer."""
    refs = [_ref("S0", 0.0, 12.0), _ref("S1", 12.8, 24.8)]
    segments = [(0.0, 12.0, "c1"), (12.8, 24.8, "c0"), (30.0, 40.0, "c2")]
    assert read_back_identities(segments, refs) == {"c1": "S0", "c0": "S1"}


def test_a_label_covering_no_reference_is_a_candidate_newcomer_not_a_guess():
    """Guessing would merge a passer-by into a participant — the one diarization
    error that cannot be repaired by reading the transcript."""
    refs = [_ref("S0", 0.0, 12.0)]
    mapping = read_back_identities([(0.0, 12.0, "c0"), (30.0, 45.0, "c7")], refs)
    assert mapping == {"c0": "S0"}
    assert "c7" not in mapping


def test_the_dominant_label_wins_a_contested_reference():
    """A reference span can be cut by the segmenter. Whoever holds most of it is the
    speaker; taking the first or last piece would be decided by a rounding error."""
    refs = [_ref("S0", 0.0, 12.0)]
    segments = [(0.0, 2.0, "noise"), (2.0, 12.0, "c3")]
    assert read_back_identities(segments, refs) == {"c3": "S0"}


def test_a_reference_nobody_covered_yields_nothing_rather_than_a_wrong_answer():
    refs = [_ref("S0", 0.0, 12.0), _ref("S1", 12.8, 24.8)]
    assert read_back_identities([(50.0, 60.0, "c9")], refs) == {}


def test_a_zero_length_reference_is_skipped():
    """An un-assembled reference has no primer span, so it anchors nothing. Treating
    its empty span as a match would attach a real voice to an empty slot."""
    assert read_back_identities([(0.0, 10.0, "c0")], [_ref("S0", 5.0, 5.0)]) == {}


# --------------------------------------------------------------------------- #
#  The detail that is easiest to get wrong
# --------------------------------------------------------------------------- #


def test_the_room_tone_gap_exceeds_the_segmenters_minimum_off_duration():
    """Below the segmentation model's `min_duration_off` (0.5 s in the measured
    configuration) the model does not cut a turn boundary at the gap at all: the
    references bleed into one another and the read-back becomes ambiguous.

    This single constant is why the primer works, and it is invisible in any
    output — a wrong value produces plausible labels that are quietly mixed up.
    """
    assert DEFAULT_GAP_S > 0.5
    assert RediarPolicy().gap_s > 0.5


def test_the_primer_charges_one_gap_per_speaker_plus_one_before_the_window():
    """The last gap is not an off-by-one: the model must also cut between the primer
    and the audio being diarized, or the final reference merges into the window."""
    policy = RediarPolicy(ref_budget_s=10.0, gap_s=1.0)
    assert policy.primer_s(speakers=3) == pytest.approx(3 * 11.0 + 1.0)
