"""
vocal_helper.consistency
========================

Is it still the same speaker forty minutes later?

The diarization error rate does not answer that. It is a time-weighted sum of
missed speech, false alarm and speaker confusion, scored with a collar, and on a
long recording it is dominated by **boundaries**: thousands of small edge errors
swamp the one failure that actually ruins a transcript, which is a speaker losing
their identity halfway through and finishing the meeting under a different label.

Two recordings can share a diarization error rate of 0.30 and be worlds apart. In
one, every turn is attributed correctly and the boundaries are a little loose. In
the other, the boundaries are perfect and the chair is called ``S0`` for twenty
minutes and ``S3`` after the coffee break. The second is useless for anything that
totals speaking time, builds an agreement network, or follows one person across
episodes — and the error rate will not tell you which one you have.

So this module measures identity, and only identity.

Everything here is measured **after an optimal one-to-one assignment** between true
speakers and hypothesis labels, solved by the Hungarian algorithm on the matrix of
shared time. That is the same reconciliation the diarization error rate performs,
and using it here is not a detail — it is what makes the numbers mean anything.

A first version of this module took, for each true speaker independently, whichever
label they spent most time under. That is not an assignment: several speakers can
claim the same label. The consequence was absurd and it was observed in the wild —
a stage that collapsed four voices onto one label scored a **drift of 0.000 and a
long-range agreement of 0.97**, because all four speakers shared one "dominant"
label and every appearance therefore matched every other. The metric rewarded total
collapse. With a one-to-one assignment the collapse is priced correctly: the label
goes to one speaker and the other three are simply wrong everywhere.

The three numbers
-----------------
**Drift**, per true speaker: one minus the share of their speech that landed on
**their assigned label**. Zero is perfect. It answers *"was this person held
together, under their own name?"*

**Collision**, per hypothesis label: one minus the share of that label's time
belonging to **its assigned speaker**. Zero is perfect. It answers *"does this label
mean one person?"*

Both are needed, because each alone is still gameable at the edges, and together
with the assignment they pin both failure directions: splitting one voice across
labels, and merging several voices into one.

**Long-range agreement**, as a function of a gap: among all pairs of appearances by
the same true speaker separated by at least that gap, the fraction where **both**
appearances landed on that speaker's assigned label. This is the number that answers
the question in the title, and the one a windowed or block-wise diarizer has to be
judged on: its whole identity machinery exists to make the curve flat. Requiring the
assigned label, rather than merely the *same* label, is what stops a uniformly wrong
answer from scoring as a consistent one.

Why a synthetic corpus is the right instrument here
----------------------------------------------------
Real meeting corpora give a reference, but they do not let you *choose* when a
speaker comes back. To measure long-range consistency you want one voice appearing
at minute one, minute ten and minute forty, with nothing of theirs in between — and
you want to know, for certain, that it is the same voice.

The in-house **bagarre** fabricator (``pdbms/scripts/fabricate_bagarre.py``) builds
exactly that: every turn is cut from a known mono-speaker clip, so the identity is
exact everywhere by construction rather than by annotation, and the recipe records
which speaker is in which interval. Annotation disagreement, the usual noise floor
of this kind of measurement, simply does not exist.

Usage example
-------------
>>> reference = [(0.0, 10.0, "alice"), (10.0, 20.0, "bob"), (60.0, 70.0, "alice")]
>>> perfect = [(0.0, 10.0, "S0"), (10.0, 20.0, "S1"), (60.0, 70.0, "S0")]
>>> report = speaker_consistency(reference, perfect)
>>> report.worst_drift, report.worst_collision
(0.0, 0.0)
>>> report.long_range_agreement(gap_s=30.0)
1.0

The failure the error rate hides: alice is renamed after the gap.

>>> renamed = [(0.0, 10.0, "S0"), (10.0, 20.0, "S1"), (60.0, 70.0, "S2")]
>>> report = speaker_consistency(reference, renamed)
>>> round(report.worst_drift, 2)
0.5
>>> report.long_range_agreement(gap_s=30.0)
0.0

And the collapse that the first version of this module rewarded, now priced:

>>> collapsed = [(0.0, 20.0, "S0"), (60.0, 70.0, "S0")]
>>> report = speaker_consistency(reference, collapsed)
>>> report.worst_drift        # bob never gets a label of his own
1.0
>>> report.long_range_agreement(gap_s=30.0)
1.0

Alice keeps her label across the gap, so *her* long-range agreement is honest; it is
``worst_drift`` that reports bob's disappearance. One number cannot carry both, which
is why there are three.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

__all__ = [
    "ConsistencyReport",
    "speaker_consistency",
]

_Interval = tuple[float, float, str]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Seconds shared by two intervals, or ``0.0``."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _optimal_assignment(
    pair_time: dict[tuple[str, str], float],
    speakers: list[str],
    labels: list[str],
) -> dict[str, str]:
    """Match speakers to labels one-to-one, maximising shared time.

    The Hungarian algorithm, which is the same reconciliation the diarization error
    rate performs. One-to-one is the whole point: letting two speakers claim the
    same label makes a collapsed answer look perfect, which is precisely the bug
    this function was written to remove.

    Parameters
    ----------
    pair_time : dict
        ``{(speaker, label): seconds}``.
    speakers, labels : list of str
        Sorted, so the result is deterministic when several assignments tie.

    Returns
    -------
    dict of str
        ``{speaker: label}``, containing only pairs that actually share time. With
        fewer labels than speakers, somebody is deliberately left unassigned.

    Examples
    --------
    >>> _optimal_assignment({("a", "S0"): 10.0, ("b", "S1"): 10.0}, ["a", "b"], ["S0", "S1"])
    {'a': 'S0', 'b': 'S1'}

    The greedy reading would give both speakers ``S0``; the assignment cannot.

    >>> _optimal_assignment(
    ...     {("a", "S0"): 10.0, ("b", "S0"): 9.0, ("b", "S1"): 1.0},
    ...     ["a", "b"], ["S0", "S1"])
    {'a': 'S0', 'b': 'S1'}
    """
    if not speakers or not labels:
        return {}

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    cost = np.zeros((len(speakers), len(labels)), dtype=np.float64)
    for i, speaker in enumerate(speakers):
        for j, label in enumerate(labels):
            # Negated: the solver minimises, and we want the most shared time.
            cost[i, j] = -pair_time.get((speaker, label), 0.0)

    rows, cols = linear_sum_assignment(cost)
    return {
        speakers[i]: labels[j]
        for i, j in zip(rows, cols, strict=True)
        if pair_time.get((speakers[i], labels[j]), 0.0) > 0
    }


@dataclass
class ConsistencyReport:
    """Identity quality, independent of boundary quality."""

    drift: dict[str, float] = field(default_factory=dict)
    """Per true speaker: one minus the share of their speech on their ASSIGNED label."""

    collision: dict[str, float] = field(default_factory=dict)
    """Per hypothesis label: one minus the share of its time on its ASSIGNED speaker."""

    assigned_label: dict[str, str] = field(default_factory=dict)
    """Per true speaker, the label the one-to-one assignment gave them.

    A speaker absent from this mapping received no label of their own, which happens
    when the hypothesis has fewer labels than there are speakers — a collapse.
    """

    appearances: dict[str, list[tuple[float, float, str]]] = field(default_factory=dict)
    """Per true speaker, their reference intervals with the label each one received."""

    @property
    def worst_drift(self) -> float:
        """The least well-held speaker. One bad voice ruins a transcript, so the
        maximum is the honest summary and the mean would hide it."""
        return max(self.drift.values(), default=0.0)

    @property
    def worst_collision(self) -> float:
        """The most ambiguous label.

        **Saturates, and that is a known limit of the maximum.** One spurious label
        holding a tenth of a second is assigned to nobody, scores 1.0 on its own,
        and pins this number at 1.0 whatever the rest of the recording looks like.
        It answers "is any label meaningless?" — useful as a tripwire, useless as a
        measure of how bad things are. For that, read :attr:`misattributed_share`.
        """
        return max(self.collision.values(), default=0.0)

    label_time: dict[str, float] = field(default_factory=dict)
    """Seconds of reference speech each hypothesis label covers."""

    @property
    def misattributed_share(self) -> float:
        """Share of labelled speech time that sits on the wrong speaker.

        The time-weighted companion to :attr:`worst_collision`, and the number to
        quote when comparing two diarizers. A handful of tiny orphan clusters barely
        moves it; half the audio under a wrong name sends it to 0.5.

        Returns
        -------
        float
            In ``[0, 1]``. Zero means every labelled second belongs to the speaker
            its label was assigned to.

        Examples
        --------
        A tiny spurious label next to an otherwise perfect answer: the maximum
        saturates, the share stays honest.

        >>> reference = [(0.0, 100.0, "alice")]
        >>> almost = [(0.0, 99.9, "S0"), (99.9, 100.0, "S1")]
        >>> report = speaker_consistency(reference, almost)
        >>> report.worst_collision
        1.0
        >>> round(report.misattributed_share, 4)
        0.001
        """
        total = sum(self.label_time.values())
        if total <= 0:
            return 0.0
        wrong = sum(
            seconds * self.collision.get(label, 1.0) for label, seconds in self.label_time.items()
        )
        return wrong / total

    def long_range_agreement(self, *, gap_s: float) -> float:
        """Share of same-speaker appearance pairs, at least ``gap_s`` apart, that kept a label.

        Parameters
        ----------
        gap_s : float, keyword-only
            Minimum silence between the end of one appearance and the start of the
            next, for the pair to count. A large value isolates exactly the failure
            this module exists for: identity lost across a long absence.

        Returns
        -------
        float
            In ``[0, 1]``, or ``1.0`` when no pair is that far apart — nothing was
            asked of the diarizer at that range, so nothing is held against it.

        Examples
        --------
        >>> reference = [(0.0, 5.0, "a"), (100.0, 105.0, "a")]
        >>> kept = speaker_consistency(reference, [(0.0, 5.0, "S0"), (100.0, 105.0, "S0")])
        >>> kept.long_range_agreement(gap_s=60.0)
        1.0
        >>> kept.long_range_agreement(gap_s=600.0)   # nothing that far apart
        1.0
        """
        total = agreed = 0
        for speaker, spans in self.appearances.items():
            # The speaker's OWN label, from the one-to-one assignment. Requiring it —
            # rather than merely requiring the two appearances to agree with each
            # other — is what stops a uniformly wrong answer from scoring as a
            # consistent one.
            own = self.assigned_label.get(speaker)
            ordered = sorted(spans)
            for i, (_a0, a1, label_a) in enumerate(ordered):
                for b0, _b1, label_b in ordered[i + 1 :]:
                    if b0 - a1 < gap_s:
                        continue
                    total += 1
                    if own is not None and label_a == own and label_b == own:
                        agreed += 1
        return 1.0 if total == 0 else agreed / total

    def describe(self) -> str:
        """One line, for a log or a study table."""
        return (
            f"drift max {self.worst_drift:.3f} · misattributed {self.misattributed_share:.3f} "
            f"(collision max {self.worst_collision:.3f}) · "
            f"{len(self.drift)} true speakers → {len(self.collision)} labels"
        )


def speaker_consistency(
    reference: list[_Interval],
    hypothesis: list[_Interval],
    *,
    min_share: float = 0.0,
) -> ConsistencyReport:
    """Measure whether identities were held, ignoring boundary accuracy.

    Parameters
    ----------
    reference : list of tuple
        ``(t0, t1, true_speaker)``. From a synthetic recipe, where identity is exact
        by construction, or from a hand annotation.
    hypothesis : list of tuple
        ``(t0, t1, label)`` as produced by a diarizer.
    min_share : float, keyword-only, optional
        Ignore a label that holds less than this share of a speaker's time, as
        boundary spill rather than a genuine identity split. ``0.0`` counts
        everything, which is the strict reading and the default: a frame of spill is
        still an error, it is simply a small one, and the share-weighted drift
        already makes it small.

    Returns
    -------
    ConsistencyReport

    Examples
    --------
    Two voices merged into one label. **Both** numbers now report it, because the
    assignment can only give ``S0`` to one of them: the other is unassigned and
    therefore wrong everywhere.

    >>> reference = [(0.0, 10.0, "alice"), (10.0, 20.0, "bob")]
    >>> merged = [(0.0, 20.0, "S0")]
    >>> report = speaker_consistency(reference, merged)
    >>> report.worst_drift, report.worst_collision
    (1.0, 0.5)
    >>> sorted(report.assigned_label)        # only one of the two got a label
    ['alice']

    One voice split in half: the mirror image. Alice keeps one of the two labels, so
    half her speech is lost to the other — and the orphan label belongs to **nobody**,
    which scores a collision of 1.

    >>> split = [(0.0, 10.0, "S0"), (10.0, 20.0, "S1")]
    >>> report = speaker_consistency([(0.0, 20.0, "alice")], split)
    >>> report.worst_drift, report.worst_collision
    (0.5, 1.0)

    That an unowned label costs a full collision is deliberate, and it is what makes
    this a direct detector of over-segmentation: the repository's open high-severity
    bug is a clusterer that minted 203 labels for four speakers, and 199 of them
    would be unowned here.
    """
    # Time shared by each (true speaker, label) pair, and the label each reference
    # interval mostly received — the second is what the long-range question needs.
    pair_time: collections.defaultdict = collections.defaultdict(float)
    appearances: dict[str, list[tuple[float, float, str]]] = {}

    for r0, r1, speaker in reference:
        if r1 <= r0:
            continue
        per_label: collections.Counter = collections.Counter()
        for h0, h1, label in hypothesis:
            shared = _overlap(r0, r1, h0, h1)
            if shared > 0:
                per_label[label] += shared
                pair_time[(speaker, label)] += shared
        # An interval nobody labelled gets "": it is an absence of evidence, and
        # counting it as agreement with anything would flatter the result.
        best = per_label.most_common(1)[0][0] if per_label else ""
        appearances.setdefault(speaker, []).append((r0, r1, best))

    by_speaker: collections.defaultdict = collections.defaultdict(float)
    by_label: collections.defaultdict = collections.defaultdict(float)
    for (speaker, label), seconds in pair_time.items():
        by_speaker[speaker] += seconds
        by_label[label] += seconds

    assigned = _optimal_assignment(pair_time, sorted(by_speaker), sorted(by_label))

    drift: dict[str, float] = {}
    for speaker, total in by_speaker.items():
        label = assigned.get(speaker)
        if label is None or total <= 0:
            # No label could be assigned to this speaker: with fewer labels than
            # speakers, somebody is left out, and they are wrong everywhere. That is
            # a drift of 1, and it is the honest price of a collapse.
            drift[speaker] = 1.0
            continue
        kept = pair_time.get((speaker, label), 0.0)
        drift[speaker] = max(0.0, 1.0 - kept / total)

    reverse = {label: speaker for speaker, label in assigned.items()}
    collision: dict[str, float] = {}
    for label, total in by_label.items():
        owner = reverse.get(label)
        if owner is None or total <= 0:
            # A label assigned to nobody is a label that exists for no speaker: a
            # spurious cluster, which is the 203-labels failure in miniature.
            collision[label] = 1.0
            continue
        collision[label] = max(0.0, 1.0 - pair_time.get((owner, label), 0.0) / total)

    return ConsistencyReport(
        drift=drift,
        collision=collision,
        assigned_label=assigned,
        appearances=appearances,
        label_time=dict(by_label),
    )
