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


# --------------------------------------------------------------------------- #
#  The primer, assembled
# --------------------------------------------------------------------------- #


def test_room_tone_is_harvested_from_the_quietest_span_not_synthesised():
    """Digital zeros are out of distribution for a segmentation model trained on
    real audio: it behaves oddly around them, exactly where a clean cut is needed."""
    from vocal_helper.rediar import harvest_room_tone

    sr = 16000
    loud = np.ones(sr, dtype=np.float32)
    ambience = np.full(sr, 0.002, dtype=np.float32)
    room = harvest_room_tone(np.concatenate([loud, ambience, loud]), sr, 0.5)
    assert len(room) == sr // 2
    assert 0.0 < float(np.abs(room).max()) < 0.01  # real ambience, not zeros


def test_room_tone_falls_back_to_zeros_only_when_there_is_not_enough_audio():
    """A degenerate case, not a choice. Saying so is what keeps it from becoming one."""
    from vocal_helper.rediar import harvest_room_tone

    room = harvest_room_tone(np.ones(10, dtype=np.float32), 16000, 0.5)
    assert len(room) == 8000
    assert float(np.abs(room).max()) == 0.0


def test_references_take_the_longest_turns_first():
    """A long turn is the one most likely to be a single voice alone. Short turns are
    where overlap and mislabelling live, and a reference built from them anchors the
    wrong person — silently, since the result is plausible either way."""
    from vocal_helper.rediar import build_references

    sr = 1000
    pcm = np.concatenate(
        [
            np.full(1000, 0.1, dtype=np.float32),  # A, 1 s — the short turn
            np.full(3000, 0.9, dtype=np.float32),  # A, 3 s — the long one
        ]
    )
    segments = [(0.0, 1.0, "A"), (1.0, 4.0, "A")]
    refs = build_references(pcm=pcm, sample_rate=sr, segments=segments, offset_s=0.0, budget_s=1.0)
    # The budget was filled from the LONG turn, so the reference carries its level.
    assert float(np.abs(refs["A"] - 0.9).max()) < 1e-6


def test_the_primer_puts_a_gap_before_every_reference_and_one_at_the_end():
    """The last gap is not an off-by-one: without it the final reference merges into
    the window that follows and anchors nothing at all."""
    from vocal_helper.rediar import assemble_primer

    sr = 1000
    room = np.zeros(800, dtype=np.float32)
    primer, refs = assemble_primer(
        {"A": np.ones(1000, dtype=np.float32), "B": np.ones(1000, dtype=np.float32)},
        room,
        sr,
    )
    assert len(primer) == 3 * 800 + 2 * 1000
    assert [r.speaker for r in refs] == ["A", "B"]
    assert refs[0].primer_t0 == pytest.approx(0.8)
    assert refs[1].primer_t0 == pytest.approx(0.8 + 1.0 + 0.8)


def test_the_window_is_reconstructed_on_the_real_timeline():
    """Concatenating voiced spans would compress time and every returned boundary
    would be wrong. The gaps are filled, not removed."""
    from vocal_helper.rediar import reconstruct_window

    sr = 1000
    room = np.full(100, 0.002, dtype=np.float32)
    speech = np.ones(500, dtype=np.float32)
    out = reconstruct_window(
        [(10.0, 10.5, speech), (12.0, 12.5, speech)], room, sr, window_start_s=10.0
    )
    assert len(out) == 2500  # 0.5 s speech + 1.5 s gap + 0.5 s speech
    assert float(out[100]) == pytest.approx(1.0)  # inside the first turn
    assert float(out[1000]) == pytest.approx(0.002, abs=1e-5)  # ambience in the gap


# --------------------------------------------------------------------------- #
#  The stage: a provisional label, then a correction
# --------------------------------------------------------------------------- #


class _FakeOnline:
    """Labels every segment with whatever it is told, in order."""

    def __init__(self, labels):
        self.labels = list(labels)

    def _label(self, segment):
        label = self.labels.pop(0) if self.labels else "S0"
        return {
            "t0": segment["t0"],
            "t1": segment["t1"],
            "sample_rate": segment["sample_rate"],
            "speaker": label,
            "pcm": segment["pcm"],
        }


class _FakeOffline:
    """Returns a canned diarization, and records what it was asked to diarize."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls: list[int] = []

    def diarize(self, pcm, sr, *, num_speakers=None, **_kw):
        self.calls.append(len(pcm))
        if not self.answers:
            raise RuntimeError("no canned answer left")
        return self.answers.pop(0)


def _segment(t0, t1, sr=1000, level=0.5):
    return {
        "t0": t0,
        "t1": t1,
        "sample_rate": sr,
        "pcm": np.full(int((t1 - t0) * sr), level, dtype=np.float32),
    }


def test_a_refresh_waits_for_its_period_and_then_runs_once():
    """A refresh per segment would spend the whole budget on repetition."""
    from vocal_helper.rediar import PeriodicOfflineDiarStage, RediarPolicy

    offline = _FakeOffline([[(0.0, 30.0, "c0")]])
    stage = PeriodicOfflineDiarStage(
        online=_FakeOnline(["S0"] * 10),
        offline=offline,
        policy=RediarPolicy(window_s=60.0, period_s=30.0, min_speaker_s=1.0),
    )
    stage._voiced = [(0.0, 10.0, np.full(10000, 0.5, dtype=np.float32))]
    assert stage.refresh(now_s=10.0, sample_rate=1000) == []
    assert offline.calls == []  # not due


def test_a_wrong_provisional_label_is_revised_and_reported():
    """The whole point of the design: the fast label is allowed to be wrong, because
    the slow one corrects it, and the correction is reported rather than hidden."""
    from vocal_helper.rediar import PeriodicOfflineDiarStage, RediarPolicy

    # The offline diarizer says the first ten seconds were one voice, not two.
    offline = _FakeOffline([[(0.0, 20.0, "c0")]])
    seen: list = []
    stage = PeriodicOfflineDiarStage(
        online=_FakeOnline([]),
        offline=offline,
        policy=RediarPolicy(window_s=60.0, period_s=20.0, min_speaker_s=1.0),
        on_revision=seen.append,
    )
    stage._published = [
        {"t0": 0.0, "t1": 10.0, "speaker": "S0"},
        {"t0": 10.0, "t1": 20.0, "speaker": "S1"},
    ]
    stage._voiced = [(0.0, 20.0, np.full(20000, 0.5, dtype=np.float32))]

    revisions = stage.refresh(now_s=20.0, sample_rate=1000)

    assert [(r.old_speaker, r.new_speaker) for r in revisions] == [("S1", "S0")]
    assert seen and seen[0] is revisions
    # The published row was corrected in place, so a later refresh does not re-report it.
    assert stage._published[1]["speaker"] == "S0"


def test_a_correct_provisional_label_produces_no_revision():
    """Reporting a revision that changes nothing would make a display flicker for
    no reason, and would train whoever reads it to ignore revisions."""
    from vocal_helper.rediar import PeriodicOfflineDiarStage, RediarPolicy

    offline = _FakeOffline([[(0.0, 20.0, "c0")]])
    stage = PeriodicOfflineDiarStage(
        online=_FakeOnline([]),
        offline=offline,
        policy=RediarPolicy(window_s=60.0, period_s=20.0, min_speaker_s=1.0),
    )
    stage._published = [{"t0": 0.0, "t1": 20.0, "speaker": "S0"}]
    stage._voiced = [(0.0, 20.0, np.full(20000, 0.5, dtype=np.float32))]
    assert stage.refresh(now_s=20.0, sample_rate=1000) == []


def test_a_failed_refresh_keeps_the_labels_it_has():
    """A diarizer that raises must cost nothing: the provisional labels are still
    the best available answer, and discarding them would make a transient failure
    worse than no refresh at all."""
    from vocal_helper.rediar import PeriodicOfflineDiarStage, RediarPolicy

    stage = PeriodicOfflineDiarStage(
        online=_FakeOnline([]),
        offline=_FakeOffline([]),  # empty: raises
        policy=RediarPolicy(window_s=60.0, period_s=20.0, min_speaker_s=1.0),
    )
    stage._published = [{"t0": 0.0, "t1": 20.0, "speaker": "S0"}]
    stage._voiced = [(0.0, 20.0, np.full(20000, 0.5, dtype=np.float32))]
    assert stage.refresh(now_s=20.0, sample_rate=1000) == []
    assert stage._published[0]["speaker"] == "S0"
    assert stage.refreshes == 0


def test_a_newcomer_is_promoted_only_past_the_floor():
    """A cluster no reference covered is either a person who just joined or an
    artefact. The floor is the only thing separating them, and promoting an artefact
    adds a speaker who never existed."""
    from vocal_helper.rediar import PeriodicOfflineDiarStage, RediarPolicy

    # "c1" speaks 0.5 s — under any sane floor. "c0" holds the rest.
    offline = _FakeOffline([[(0.0, 19.5, "c0"), (19.5, 20.0, "c1")]])
    stage = PeriodicOfflineDiarStage(
        online=_FakeOnline([]),
        offline=offline,
        policy=RediarPolicy(window_s=60.0, period_s=20.0, min_speaker_s=5.0),
    )
    stage._published = [{"t0": 19.5, "t1": 20.0, "speaker": "S0"}]
    stage._voiced = [(0.0, 20.0, np.full(20000, 0.5, dtype=np.float32))]
    stage.refresh(now_s=20.0, sample_rate=1000)
    # No S1 invented from half a second of audio.
    assert all(r.new_speaker != "S1" for r in stage.revisions)


async def _drive(stage, segments, sr=1000):
    """Push segments through the stage's queue contract and collect the output."""
    import asyncio

    inbox: asyncio.Queue = asyncio.Queue()
    outbox: asyncio.Queue = asyncio.Queue()
    for seg in segments:
        inbox.put_nowait(seg)
    inbox.put_nowait(None)
    await stage.run(inbox, outbox)
    out = []
    while not outbox.empty():
        out.append(outbox.get_nowait())
    return out


def test_the_stage_is_a_drop_in_and_emits_each_segment_once():
    """The audio path stays once-only: a revision carries labels, never audio, so
    nothing downstream re-transcribes anything."""
    import asyncio

    from vocal_helper.rediar import PeriodicOfflineDiarStage, RediarPolicy

    offline = _FakeOffline([[(0.0, 30.0, "c0")], [(0.0, 30.0, "c0")]])
    stage = PeriodicOfflineDiarStage(
        online=_FakeOnline(["S0", "S0", "S0"]),
        offline=offline,
        policy=RediarPolicy(window_s=60.0, period_s=15.0, min_speaker_s=1.0),
    )
    out = asyncio.run(
        _drive(stage, [_segment(0.0, 10.0), _segment(10.0, 20.0), _segment(20.0, 30.0)])
    )
    assert out[-1] is None
    emitted = [o for o in out if o is not None]
    assert len(emitted) == 3  # one per segment, never twice
    assert [(e["t0"], e["t1"]) for e in emitted] == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]


def test_the_stream_closing_forces_one_last_refresh():
    """The final window has never been re-diarized, and it is the part the reader is
    still looking at. Ending without it would leave the worst labels on screen."""
    import asyncio

    from vocal_helper.rediar import PeriodicOfflineDiarStage, RediarPolicy

    offline = _FakeOffline([[(0.0, 10.0, "c0")]])
    stage = PeriodicOfflineDiarStage(
        online=_FakeOnline(["S0"]),
        offline=offline,
        policy=RediarPolicy(window_s=60.0, period_s=3600.0, min_speaker_s=1.0),
    )
    asyncio.run(_drive(stage, [_segment(0.0, 10.0)]))
    assert stage.refreshes == 1  # the period never elapsed, yet the close forced it


def test_old_audio_is_dropped_so_a_live_session_stays_bounded():
    """A live session is unbounded; keeping every voiced segment would make this
    stage the memory leak that the bounded window exists to avoid."""
    import asyncio

    from vocal_helper.rediar import PeriodicOfflineDiarStage, RediarPolicy

    offline = _FakeOffline([[(0.0, 10.0, "c0")] for _ in range(20)])
    stage = PeriodicOfflineDiarStage(
        online=_FakeOnline(["S0"] * 20),
        offline=offline,
        policy=RediarPolicy(window_s=20.0, period_s=10.0, min_speaker_s=1.0),
    )
    segments = [_segment(float(i * 10), float(i * 10 + 10)) for i in range(12)]
    asyncio.run(_drive(stage, segments))
    kept = sum(v[1] - v[0] for v in stage._voiced)
    assert kept <= 20.0 + 10.0 + 10.0  # window + period + the segment just added


# --------------------------------------------------------------------------- #
#  Reachable from the pipeline, by asking for it
# --------------------------------------------------------------------------- #


def test_the_pipeline_keeps_the_online_stage_by_default():
    """The default must not change under anybody: `discussion-helper` wants a label
    in half a second and no retroactive correction, which is the opposite priority."""
    import vocal_helper as voh
    from vocal_helper.diar import OnlineDiarStage

    async def source():
        yield {"t0": 0.0, "sample_rate": 16000, "pcm": np.zeros(320, dtype=np.float32)}

    pipeline = voh.Pipeline(source=source, config=voh.PipelineConfig(llm=None))
    assert isinstance(pipeline._diar, OnlineDiarStage)


def test_asking_for_rediar_composes_both_stages_and_keeps_the_callback():
    """The quality path is reachable by configuration, and the revision callback
    survives the trip — a display that never hears about revisions keeps showing
    labels it has been told are wrong."""
    import vocal_helper as voh
    from vocal_helper.rediar import PeriodicOfflineDiarStage

    seen: list = []

    def recolour(revisions):
        seen.extend(revisions)

    async def source():
        yield {"t0": 0.0, "sample_rate": 16000, "pcm": np.zeros(320, dtype=np.float32)}

    pipeline = voh.Pipeline(
        source=source,
        config=voh.PipelineConfig(llm=None, rediar={"period_s": 45.0, "on_revision": recolour}),
    )
    stage = pipeline._diar
    assert isinstance(stage, PeriodicOfflineDiarStage)
    assert stage.policy.period_s == 45.0
    assert stage.on_revision is recolour
    # The online stage is composed, not discarded: it is what answers in half a second.
    assert stage.online is pipeline._online_diar


def test_the_offline_backend_comes_from_the_policy_not_from_the_online_config():
    """`config.diar["backend"]` names an EMBEDDER for the online stage; the policy's
    names a whole offline pipeline. Conflating them would quietly ask for a
    diarizer that does not exist."""
    import vocal_helper as voh

    async def source():
        yield {"t0": 0.0, "sample_rate": 16000, "pcm": np.zeros(320, dtype=np.float32)}

    pipeline = voh.Pipeline(
        source=source,
        config=voh.PipelineConfig(
            llm=None, diar={"backend": "sherpa"}, rediar={"backend": "pyannote"}
        ),
    )
    assert pipeline._diar.offline.backend == "pyannote"
    assert pipeline._online_diar.backend == "sherpa"


# ----- la relecture est une bijection ---------------------------------
#
# Une version antérieure prenait, pour chaque référence, l'étiquette qui la
# couvrait le plus, et écrivait ``out[étiquette] = locuteur``. Dès que le moteur
# fusionnait deux locuteurs de l'amorçage en une seule grappe, la seconde
# écriture écrasait la première : un locuteur connu perdait son étiquette en
# silence et était rapporté absent. L'affectation optimale ne peut pas le faire.


def _refs(*spans):
    """Des références occupant ces créneaux d'amorçage."""
    import numpy as np

    from vocal_helper.rediar import SpeakerReference

    silence = np.zeros(1, dtype=np.float32)
    return [SpeakerReference(f"S{i}", silence, t0, t1) for i, (t0, t1) in enumerate(spans)]


def test_a_merged_label_claims_one_reference_and_leaves_the_other_unplaced():
    from vocal_helper.rediar import read_back_identities

    refs = _refs((0.0, 12.0), (12.8, 24.8))
    # Une seule grappe couvre les deux créneaux : le moteur a fusionné.
    got = read_back_identities([(0.0, 24.8, "c0")], refs)
    assert len(got) == 1, f"une étiquette ne peut pas porter deux identités : {got}"
    assert set(got) == {"c0"}


def test_no_known_speaker_is_silently_dropped_when_labels_suffice():
    from vocal_helper.rediar import read_back_identities

    refs = _refs((0.0, 12.0), (12.8, 24.8), (25.6, 37.6))
    segs = [(0.0, 12.0, "a"), (12.8, 24.8, "b"), (25.6, 37.6, "c")]
    got = read_back_identities(segs, refs)
    assert sorted(got.values()) == ["S0", "S1", "S2"]


def test_the_assignment_maximises_total_shared_time():
    """Le glouton pouvait choisir localement bien et globalement mal."""
    from vocal_helper.rediar import read_back_identities

    refs = _refs((0.0, 10.0), (10.0, 20.0))
    # « a » couvre un peu plus de S0 que « b », mais le lui donner force « b »
    # sur S0 aussi ; l'affectation optimale préfère la somme la plus grande.
    segs = [(0.0, 6.0, "a"), (6.0, 10.0, "b"), (10.0, 20.0, "a")]
    got = read_back_identities(segs, refs)
    assert got == {"b": "S0", "a": "S1"}, got


def test_min_coverage_refuses_a_label_that_merely_brushed_a_reference():
    from vocal_helper.rediar import read_back_identities

    refs = _refs((0.0, 12.0))
    segs = [(11.9, 12.0, "c0")]  # 0,1 s sur une référence de 12 s
    assert read_back_identities(segs, refs) == {"c0": "S0"}
    assert read_back_identities(segs, refs, min_coverage=0.5) == {}


def test_a_reference_with_no_duration_is_ignored():
    from vocal_helper.rediar import read_back_identities

    refs = _refs((5.0, 5.0))
    assert read_back_identities([(0.0, 10.0, "c0")], refs) == {}


def test_no_segments_at_all_places_nobody():
    from vocal_helper.rediar import read_back_identities

    assert read_back_identities([], _refs((0.0, 12.0))) == {}
