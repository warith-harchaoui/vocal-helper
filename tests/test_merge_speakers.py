"""
Tests for :func:`vocal_helper.merge_speakers` — repairing an over-estimated speaker count.

Module summary
--------------
Every diarizer measured in this suite fails the same way on real panel audio: it finds too
many speakers, not too few. These tests pin the repair, and in particular pin the property
that motivated changing it — a merge that is correct at one threshold and catastrophic one
notch above it is not a usable default, however good its best case looks.

Model-free: the function takes vectors and returns labels, so nothing here loads torch,
onnxruntime or any model.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import numpy as np
import pytest

from vocal_helper import merge_speakers
from vocal_helper.diar import MERGE_LINKAGE, MERGE_THRESHOLD

DIM = 64


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _speaker_basis(n: int, seed: int = 0) -> list[np.ndarray]:
    """*n* mutually near-orthogonal direction vectors — one per distinct person."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((DIM, n)))[0].T
    return [_unit(basis[i]) for i in range(n)]


def _jitter(centre: np.ndarray, amount: float, rng) -> np.ndarray:
    """Perturb *centre* by a vector of norm *amount*, so the knob means the same at any DIM.

    Scaling raw per-dimension noise would make the same number mild in 8 dimensions and
    overwhelming in 512 — the perturbation's norm grows as sqrt(DIM) — which is exactly how
    a fixture ends up testing noise instead of the thing it was written for.
    """
    n = rng.standard_normal(DIM).astype(np.float32)
    return _unit(centre + amount * (n / np.linalg.norm(n)))


def _fragments(basis, per_speaker: int, jitter: float, seed: int = 1):
    """Simulate over-clustering: each speaker split into *per_speaker* noisy labels."""
    rng = np.random.default_rng(seed)
    emb, labels, truth = [], [], []
    for s, centre in enumerate(basis):
        for k in range(per_speaker):
            emb.append(_jitter(centre, jitter, rng))
            labels.append(f"P{s}_{k}")  # a distinct provisional label per fragment
            truth.append(f"S{s}")
    return emb, labels, truth


def _groups(labels: list[str]) -> set[frozenset[int]]:
    """The PARTITION the labels describe, with names discarded — merging is about grouping."""
    by: dict[str, set[int]] = {}
    for i, lab in enumerate(labels):
        by.setdefault(lab, set()).add(i)
    return {frozenset(v) for v in by.values()}


# --------------------------------------------------------------- the core behaviour


def test_fragments_of_one_person_are_put_back_together():
    basis = _speaker_basis(4)
    emb, labels, truth = _fragments(basis, per_speaker=3, jitter=0.25)
    assert len({*labels}) == 12  # over-clustered 4 people into 12
    out = merge_speakers(emb, labels)
    assert len({*out}) == 4
    assert _groups(out) == _groups(truth)


def test_an_already_correct_labelling_is_left_alone():
    """The repair must be a no-op when there is nothing to repair."""
    basis = _speaker_basis(4)
    emb, _labels, truth = _fragments(basis, per_speaker=1, jitter=0.1)
    out = merge_speakers(emb, truth)
    assert _groups(out) == _groups(truth)


def test_distinct_speakers_are_never_welded_together():
    """The failure that matters: merging two real people is unrecoverable downstream."""
    basis = _speaker_basis(6)
    emb, labels, truth = _fragments(basis, per_speaker=2, jitter=0.2)
    out = merge_speakers(emb, labels)
    assert len({*out}) == 6
    assert _groups(out) == _groups(truth)


# ------------------------------------------------- the reason the linkage changed


def test_single_linkage_has_a_cliff_and_complete_linkage_does_not():
    """The measured defect behind the new default.

    Single linkage merges two groups when ANY one member of each is close, and the weld is
    transitive — so a chain of ambiguous fragments collapses unrelated speakers. On AMI that
    showed up as recovering the right speakers at 0.30 and collapsing an entire meeting into
    one cluster at 0.40. A default sitting on that edge is one bad audio file from ruin.
    """
    rng = np.random.default_rng(7)
    a, b = _speaker_basis(2)
    emb = [_jitter(a, 0.2, rng) for _ in range(4)] + [_jitter(b, 0.2, rng) for _ in range(4)]
    labels = [f"A{i}" for i in range(4)] + [f"B{i}" for i in range(4)]

    # The bridge: overlapped speech, where both people are in one embedding. Real audio
    # produces these constantly, and they are what single linkage walks across.
    for i, w in enumerate((0.35, 0.5, 0.65)):
        emb.append(_unit((1 - w) * a + w * b))
        labels.append(f"OVERLAP{i}")

    single = len({*merge_speakers(emb, labels, linkage="single", min_cluster_size=1)})
    complete = len({*merge_speakers(emb, labels, linkage="complete", min_cluster_size=1)})

    assert single == 1, f"single linkage should chain A-overlap-B into one, got {single}"
    assert complete > 1, f"complete linkage should keep A and B apart, got {complete}"


def test_the_shipped_default_is_the_robust_linkage():
    assert MERGE_LINKAGE == "complete"
    assert 0.0 < MERGE_THRESHOLD < 1.0


# --------------------------------------------------------------- floors and ceilings


def test_min_speakers_is_a_floor_the_merge_cannot_cross():
    """A known roster proves those people are distinct; merging past it destroys that."""
    basis = _speaker_basis(5)
    emb, labels, _ = _fragments(basis, per_speaker=2, jitter=0.2)
    # A threshold absurd enough to fold everyone together...
    assert len({*merge_speakers(emb, labels, threshold=1.9)}) == 1
    # ...is overruled by the floor.
    assert len({*merge_speakers(emb, labels, threshold=1.9, min_speakers=5)}) == 5


def test_max_speakers_caps_the_result():
    basis = _speaker_basis(6)
    emb, labels, _ = _fragments(basis, per_speaker=2, jitter=0.2)
    assert len({*merge_speakers(emb, labels, threshold=0.01, max_speakers=3)}) == 3


def test_floor_is_clamped_to_what_exists():
    """Asking for more speakers than there are labels must not invent any."""
    basis = _speaker_basis(2)
    emb, labels, _ = _fragments(basis, per_speaker=1, jitter=0.1)
    assert len({*merge_speakers(emb, labels, threshold=1.9, min_speakers=99)}) == 2


# --------------------------------------------------------------- pruning and weighting


def test_a_micro_cluster_is_absorbed_into_its_nearest_real_speaker():
    """Overlap, laughter and jingles mint one-off speakers; they must not survive."""
    basis = _speaker_basis(2)
    emb, labels, _ = _fragments(basis, per_speaker=3, jitter=0.15)
    emb.append(_unit(basis[0] + 0.3 * basis[1]))  # one ambiguous outlier, leaning to S0
    labels.append("JUNK")
    out = merge_speakers(emb, labels, min_cluster_size=2)
    assert len({*out}) == 2
    assert out[-1] == out[0], "the outlier should join the speaker it most resembles"


def test_duration_weighting_lets_the_long_turn_decide():
    """A crowd of 1 s backchannels must not outvote 30 s of actual speech."""
    basis = _speaker_basis(2)
    long_turn = _unit(basis[0])
    shorts = [_unit(basis[0] + 0.9 * basis[1]) for _ in range(6)]
    emb = [long_turn, *shorts, _unit(basis[1])]
    labels = ["A"] * 7 + ["B"]
    durations = [30.0] + [1.0] * 6 + [30.0]
    weighted = merge_speakers(emb, labels, durations=durations, threshold=0.2)
    assert len({*weighted}) == 2, "A and B are different people and must stay apart"


# --------------------------------------------------------------- contracts and edges


def test_unembedded_segments_keep_their_label_untouched():
    basis = _speaker_basis(2)
    emb, labels, _ = _fragments(basis, per_speaker=2, jitter=0.15)
    emb.insert(2, None)
    labels.insert(2, "S?")
    out = merge_speakers(emb, labels)
    assert out[2] == "S?"


def test_a_single_label_is_returned_unchanged():
    emb = [_unit(np.ones(DIM, dtype=np.float32))] * 3
    assert merge_speakers(emb, ["S0"] * 3) == ["S0"] * 3


def test_empty_input_is_not_an_error():
    assert merge_speakers([], []) == []


def test_mismatched_lengths_fail_loudly():
    emb = [_unit(np.ones(DIM, dtype=np.float32))]
    with pytest.raises(ValueError, match="differ in length"):
        merge_speakers(emb, ["S0", "S1"])
    with pytest.raises(ValueError, match="durations must align"):
        merge_speakers(emb, ["S0"], durations=[1.0, 2.0])


def test_unknown_linkage_is_named_rather_than_failing_later():
    emb = [_unit(np.ones(DIM, dtype=np.float32))] * 2
    with pytest.raises(ValueError, match="unknown linkage"):
        merge_speakers(emb, ["S0", "S1"], linkage="ward")


def test_output_labels_are_compact_and_first_appearance_ordered():
    basis = _speaker_basis(3)
    emb, labels, _ = _fragments(basis, per_speaker=2, jitter=0.15)
    out = merge_speakers(emb, labels)
    assert out[0] == "S0"
    assert {*out} == {f"S{i}" for i in range(len({*out}))}


def test_result_is_deterministic():
    basis = _speaker_basis(4)
    emb, labels, _ = _fragments(basis, per_speaker=3, jitter=0.3)
    assert merge_speakers(emb, labels) == merge_speakers(emb, labels)


# --------------------------------------------------------------- channel centering


def test_centering_survives_a_shared_channel_that_swamps_the_speakers():
    """The case centering exists for: one mic, one room, everybody through it.

    A large common component added to every voiceprint pushes all of them into the same
    corner of the sphere, so in the raw space even different people look alike and the merge
    folds them together. Removing the per-recording mean restores the difference.
    """
    basis = _speaker_basis(4)
    rng = np.random.default_rng(3)
    channel = _unit(rng.standard_normal(DIM).astype(np.float32))
    emb, labels = [], []
    for s, centre in enumerate(basis):
        for k in range(2):
            # The channel dominates: 3x the weight of the speaker's own direction.
            emb.append(_unit(_jitter(centre, 0.15, rng) + 3.0 * channel))
            labels.append(f"P{s}_{k}")

    assert len({*merge_speakers(emb, labels, centre=False)}) < 4, (
        "fixture is not actually channel-dominated; the test would prove nothing"
    )
    assert len({*merge_speakers(emb, labels, centre=True)}) == 4


def test_centering_uses_one_vote_per_speaker_not_per_segment():
    """A dominant speaker must not become the thing that gets subtracted.

    Centering on a segment-mean would lean towards whoever talks most, so the "shared
    channel" removed would be mostly that person — which is how a host gets erased and
    everyone else gets compared to his residue.
    """
    basis = _speaker_basis(3)
    rng = np.random.default_rng(11)
    emb, labels, durations = [], [], []
    for k in range(2):  # the host: many short segments
        for _ in range(20):
            emb.append(_jitter(basis[0], 0.15, rng))
            labels.append(f"HOST_{k}")
            durations.append(2.0)
    for s in (1, 2):  # two guests, few segments
        for k in range(2):
            for _ in range(3):
                emb.append(_jitter(basis[s], 0.15, rng))
                labels.append(f"G{s}_{k}")
                durations.append(2.0)
    out = merge_speakers(emb, labels, durations=durations, centre=True)
    assert len({*out}) == 3, f"host dominance broke the merge: {len({*out})} speakers"


def test_centre_false_still_works_at_its_own_band():
    """Turning centering off is supported — the threshold just means something else."""
    basis = _speaker_basis(4)
    emb, labels, truth = _fragments(basis, per_speaker=3, jitter=0.25)
    out = merge_speakers(emb, labels, centre=False, threshold=0.35)
    assert _groups(out) == _groups(truth)


def test_noise_cannot_be_promoted_to_a_speaker_by_merging_with_other_noise():
    """Why pruning happens BEFORE merging, not after.

    Two junk labels — a laugh and a jingle — that happen to resemble each other will merge
    into a cluster of two, and so pass the very size test meant to remove them. Noise gets
    promoted to a speaker by being merged with other noise. Pruning first means only labels
    with real support ever take part in the merge.
    """
    basis = _speaker_basis(2)
    rng = np.random.default_rng(5)
    emb, labels = [], []
    for s, centre in enumerate(basis):
        for _ in range(6):
            emb.append(_jitter(centre, 0.15, rng))
            labels.append(f"REAL{s}")
    # Two singleton junk labels that resemble each other far more than any real speaker.
    junk = _unit(rng.standard_normal(DIM).astype(np.float32))
    emb.append(junk)
    labels.append("JUNK_A")
    emb.append(_jitter(junk, 0.05, rng))
    labels.append("JUNK_B")

    out = merge_speakers(emb, labels, min_cluster_size=2)
    assert len({*out}) == 2, f"junk survived as its own speaker: {len({*out})} clusters"


def test_the_centre_is_not_dragged_around_by_one_off_outliers():
    """Pruning first also keeps noise out of the channel estimate."""
    basis = _speaker_basis(3)
    rng = np.random.default_rng(9)
    emb, labels = [], []
    for s, centre in enumerate(basis):
        for _ in range(5):
            emb.append(_jitter(centre, 0.15, rng))
            labels.append(f"REAL{s}")
    for j in range(25):  # a crowd of singleton outliers
        emb.append(_unit(rng.standard_normal(DIM).astype(np.float32)))
        labels.append(f"NOISE{j}")
    out = merge_speakers(emb, labels, min_cluster_size=2)
    assert len({*out}) == 3
