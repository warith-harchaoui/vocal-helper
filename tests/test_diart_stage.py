"""diart's snapshot semantics, converted to a stream — the part with the reasoning in it.

diart re-emits its entire running annotation on every step. A list can be rebuilt from each
snapshot; a queue cannot, because a consumer cannot un-receive what it already got. These
cover that conversion without diart, without audio and without an event loop.
"""

from vocal_helper.diar import DiartDiarStage, snapshot_to_delta


def test_the_first_snapshot_is_entirely_new() -> None:
    sent = {}
    out, revisions = snapshot_to_delta([(0.0, 1.0, "a"), (1.0, 2.0, "b")], sent)
    assert out == [(0.0, 1.0, "a"), (1.0, 2.0, "b")] and revisions == 0


def test_re_sending_the_same_snapshot_emits_nothing() -> None:
    """The whole point: appending every snapshot would deliver each segment once per step."""
    sent = {}
    snap = [(0.0, 1.0, "a")]
    snapshot_to_delta(snap, sent)
    out, revisions = snapshot_to_delta(snap, sent)
    assert out == [] and revisions == 0


def test_a_segment_that_grew_is_re_emitted_and_counted_as_a_revision() -> None:
    """A segment that extended is news; counting it separately is what makes flicker visible."""
    sent = {}
    snapshot_to_delta([(0.0, 1.0, "a")], sent)
    out, revisions = snapshot_to_delta([(0.0, 1.8, "a")], sent)
    assert out == [(0.0, 1.8, "a")] and revisions == 1


def test_a_genuinely_new_segment_is_not_a_revision() -> None:
    sent = {}
    snapshot_to_delta([(0.0, 1.0, "a")], sent)
    out, revisions = snapshot_to_delta([(0.0, 1.0, "a"), (2.0, 3.0, "b")], sent)
    assert out == [(2.0, 3.0, "b")] and revisions == 0


def test_the_same_start_under_another_speaker_is_a_different_segment() -> None:
    sent = {}
    snapshot_to_delta([(0.0, 1.0, "a")], sent)
    out, _ = snapshot_to_delta([(0.0, 1.0, "a"), (0.0, 1.0, "b")], sent)
    assert out == [(0.0, 1.0, "b")]


# --- the stage ----------------------------------------------------------------------------- #


def test_labels_are_mapped_to_this_package_s_stable_ids() -> None:
    stage = DiartDiarStage()
    out = stage.consume_snapshot([(0.0, 1.0, "speaker_7"), (1.0, 2.0, "speaker_3")])
    assert [s["speaker"] for s in out] == ["S0", "S1"]
    # Stable for the rest of the session: the same diart label keeps its id.
    again = stage.consume_snapshot([(5.0, 6.0, "speaker_7")])
    assert again[0]["speaker"] == "S0"


def test_two_sessions_do_not_share_a_numbering() -> None:
    """A class-level mapping would leak one recording's speakers into the next."""
    first = DiartDiarStage()
    first.consume_snapshot([(0.0, 1.0, "x"), (1.0, 2.0, "y")])
    second = DiartDiarStage()
    assert second.consume_snapshot([(0.0, 1.0, "z")])[0]["speaker"] == "S0"


def test_the_stage_counts_its_own_revisions() -> None:
    """The honest price of a 500 ms first label — read it next to the latency, never alone."""
    stage = DiartDiarStage()
    stage.consume_snapshot([(0.0, 1.0, "a")])
    stage.consume_snapshot([(0.0, 2.0, "a")])
    stage.consume_snapshot([(0.0, 2.5, "a")])
    assert stage.revisions == 2


def test_emitted_segments_carry_the_sample_rate_downstream_needs() -> None:
    stage = DiartDiarStage(sample_rate=8_000)
    assert stage.consume_snapshot([(0.0, 1.0, "a")])[0]["sample_rate"] == 8_000
