"""
vocal_helper.rediar
===================

Offline-quality diarization on a live stream, at a cost that does not grow.

The online diarizer is a latency-bound approximation: it embeds each voiced
segment as it arrives and clusters by cosine distance, which cannot model
overlapping speech and so sits three to four times above the offline error rate
(measured, :mod:`vocal_helper.router`: **0.586 online against 0.122 offline**).
When the speaker labels matter as much as the words, that is not good enough, and
no better online clusterer fixes it — the limitation is the latency, not the
clusterer.

What fixes it is running the **offline** diarizer, periodically, on a bounded
window, and correcting the labels already shown. The offline diarizer is cheap per
second of audio; what makes it unaffordable live is re-running it on everything.

Not a new algorithm: the live expression of an existing one
-----------------------------------------------------------
The identity strategy here is **ported from the native core's block-wise
diarizer** (``core/nh-core/src/blockwise.rs``, same author), which solved this for
long recordings. That code earned three details this module would otherwise have
got wrong, and each of them is the difference between working and almost working:

1. **The primer is a per-speaker budget, not a lump.** Each already-known speaker
   contributes a few seconds of *their own* harvested audio. A single block of
   "earlier audio" would simply miss whoever had not spoken yet, and their label
   would then drift at every refresh.
2. **The pieces are separated by harvested room tone, not silence.** Real
   low-level ambience from the same recording, and the gap must exceed the
   segmentation model's ``min_duration_off`` (0.5 s in the measured configuration,
   hence a 0.8 s default) or the model does not cut a turn boundary there at all.
   Digital silence is worse than useless: it is out of distribution, and the model
   behaves oddly around it.
3. **Identity is recovered by read-back, not by matching.** The primer is prepended
   and ``[primer | window]`` is diarized in one call, so the engine's *own*
   clustering puts each window turn in the same cluster as the matching reference.
   Then you simply read which local label covered each reference span. No
   embeddings, no centroids, no similarity threshold of our own — and exact,
   where an overlap heuristic is approximate.

Why the obvious version does not hold
-------------------------------------
"Re-diarize the whole buffer every thirty seconds" is the natural design. Its
instantaneous cost is ``backend_rtf × buffer_s / period_s``, and ``buffer_s``
grows with the session: at 0.067 and a thirty-second period, one hour reaches
``0.067 × 3600 / 30 ≈ 8.0``. It works beautifully for three minutes and collapses
after twenty. :meth:`RediarPolicy.unbounded_rtf` exists so that number can be
looked at rather than assumed away.

Bounded, the cost is constant in the session length:

    rtf = backend_rtf × (window_s + primer_s(speakers)) / period_s

**And it grows with the speaker count**, because the primer does — mildly, at the
default settings: 12.8 s of primer per speaker against a 180 s window, so each
extra voice costs 0.014 of real-time factor. The default policy carries
**eighteen** speakers on a Metal machine before the budget breaks, which is well
past any real panel. The dependency still belongs in the model rather than in a
footnote: speakers arrive during a session, and a budget that silently assumed the
room it started with is a budget that fails in the second half.

Usage example
-------------
>>> policy = RediarPolicy()
>>> policy.window_s, policy.ref_budget_s, policy.gap_s, policy.period_s
(180.0, 12.0, 0.8, 60.0)
>>> round(policy.primer_s(speakers=4), 1)
52.0
>>> round(policy.stage_rtf(0.067, speakers=4), 3)
0.259
>>> policy.max_speakers_within(asr_rtf=0.033, backend_rtf=0.067)
18
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_GAP_S",
    "DEFAULT_MIN_SPEAKER_S",
    "DEFAULT_PERIOD_S",
    "DEFAULT_REF_BUDGET_S",
    "DEFAULT_WINDOW_S",
    "RediarPolicy",
    "SpeakerReference",
    "plan_refresh",
    "read_back_identities",
]

DEFAULT_WINDOW_S = 180.0
"""Trailing audio per refresh. Balances per-call clustering cost against primer overhead."""

DEFAULT_REF_BUDGET_S = 12.0
"""Per-speaker reference audio: enough for a strong embedder to cluster it, cheap enough to repeat."""

DEFAULT_GAP_S = 0.8
"""Room tone between pieces. **Must exceed the segmentation model's ``min_duration_off``** (0.5 s)."""

DEFAULT_MIN_SPEAKER_S = 15.0
"""In-window speech before a new cluster becomes a speaker. The guard against over-segmentation."""

DEFAULT_PERIOD_S = 60.0
"""New audio between refreshes. Also the worst case for how long a wrong label stays on screen."""


@dataclass(frozen=True)
class RediarPolicy:
    """How often to re-diarize, over how much audio, and what that costs.

    Parameters
    ----------
    window_s : float
        Trailing audio re-diarized at each refresh.
    ref_budget_s : float
        Audio harvested **per already-known speaker** for the primer.
    gap_s : float
        Room tone inserted between primer pieces and before the window. Below the
        segmentation model's ``min_duration_off`` the model does not cut there, the
        references bleed into each other, and the read-back becomes ambiguous.
    period_s : float
        New audio accumulated between refreshes.
    min_speaker_s : float
        In-window speech a new cluster must clear to be promoted to a speaker.
    backend : str
        The **offline** diarizer. ``"pyannote"`` is the measured default at an error
        rate of 0.122; ``"sherpa"`` is the torch-free option at 0.174.
    """

    window_s: float = DEFAULT_WINDOW_S
    ref_budget_s: float = DEFAULT_REF_BUDGET_S
    gap_s: float = DEFAULT_GAP_S
    period_s: float = DEFAULT_PERIOD_S
    min_speaker_s: float = DEFAULT_MIN_SPEAKER_S
    backend: str = "pyannote"

    def primer_s(self, *, speakers: int) -> float:
        """Primer length for a given number of known speakers.

        One reference plus one gap each, and one more gap before the window so the
        model also cuts between the primer and the audio being diarized.

        Examples
        --------
        >>> RediarPolicy().primer_s(speakers=0)
        0.0
        >>> round(RediarPolicy().primer_s(speakers=1), 1)
        13.6
        >>> round(RediarPolicy().primer_s(speakers=6), 1)
        77.6
        """
        if speakers <= 0:
            return 0.0
        return speakers * (self.ref_budget_s + self.gap_s) + self.gap_s

    def stage_rtf(self, backend_rtf: float, *, speakers: int) -> float:
        """What this stage costs as a real-time factor, at a given speaker count.

        Constant in the session length. **Not** constant in the speaker count, and
        that dependency is the one a live budget has to carry: speakers arrive
        during a session.

        Examples
        --------
        >>> round(RediarPolicy().stage_rtf(0.067, speakers=2), 3)
        0.23
        >>> round(RediarPolicy().stage_rtf(0.067, speakers=8), 3)
        0.316
        """
        return backend_rtf * (self.window_s + self.primer_s(speakers=speakers)) / self.period_s

    @staticmethod
    def unbounded_rtf(*, buffer_s: float, period_s: float, backend_rtf: float) -> float:
        """What re-diarizing the **whole** buffer would cost, for comparison.

        A function rather than a comment, because the number is the argument: small
        enough to look fine in a demo, impossible in use. That is exactly how this
        design error survives a review.

        Examples
        --------
        >>> RediarPolicy.unbounded_rtf(buffer_s=180.0, period_s=60.0, backend_rtf=0.067)
        0.201
        >>> round(RediarPolicy.unbounded_rtf(buffer_s=3600.0, period_s=60.0, backend_rtf=0.067), 1)
        4.0
        """
        return backend_rtf * buffer_s / period_s

    def holds(
        self,
        *,
        asr_rtf: float,
        backend_rtf: float,
        speakers: int = 6,
        vad_rtf: float = 0.010,
        margin: float = 2.0,
    ) -> bool:
        """Whether the whole system stays under a real-time factor of 1.

        Assumes the worst case for placement — transcription and diarization on the
        **same** compute unit, so their costs add. That is the truth on a
        single-GPU machine, and assuming otherwise would be optimism dressed as
        arithmetic.

        Parameters
        ----------
        speakers : int, keyword-only, optional
            Expected speaker count. Six by default, which is a panel: budget for
            the room you might get, not the room you started with.
        margin : float, keyword-only, optional
            Required headroom. Two by default: crossing 1 is not a graceful
            degradation but a delay that grows for the rest of the session.

        Examples
        --------
        >>> RediarPolicy().holds(asr_rtf=0.033, backend_rtf=0.067)
        True
        >>> RediarPolicy(period_s=10.0).holds(asr_rtf=0.033, backend_rtf=0.067)
        False
        """
        from vocal_helper.accel import StageLoad, live_budget

        budget = live_budget(
            [
                StageLoad("vad", "cpu", vad_rtf),
                StageLoad("asr", "gpu", asr_rtf),
                StageLoad("rediar", "gpu", self.stage_rtf(backend_rtf, speakers=speakers)),
            ]
        )
        return budget.holds(margin=margin)

    def max_speakers_within(
        self,
        *,
        asr_rtf: float,
        backend_rtf: float,
        margin: float = 2.0,
        ceiling: int = 24,
    ) -> int:
        """How many speakers this policy can carry before the budget breaks.

        The number to know **before** a session, not during: when the answer is
        four and a fifth person joins, the pipeline starts falling behind, and the
        honest response is to widen the period rather than to discover it live.

        Returns
        -------
        int
            The largest speaker count that still holds. ``0`` when even one speaker
            does not fit, which means the policy is wrong for this machine.

        Examples
        --------
        >>> RediarPolicy().max_speakers_within(asr_rtf=0.033, backend_rtf=0.067)
        18

        Without a GPU, transcription costs five times more and the room shrinks —
        still a real panel, but no longer unlimited:

        >>> RediarPolicy().max_speakers_within(asr_rtf=0.164, backend_rtf=0.067)
        9

        Refresh too often and nothing fits, because the primer is re-diarized every
        time: a twenty-second period spends more on repeating itself than on new audio.

        >>> RediarPolicy(period_s=20.0).max_speakers_within(asr_rtf=0.033, backend_rtf=0.067)
        0
        """
        best = 0
        for count in range(1, ceiling + 1):
            if self.holds(asr_rtf=asr_rtf, backend_rtf=backend_rtf, speakers=count, margin=margin):
                best = count
            else:
                break
        return best


@dataclass
class SpeakerReference:
    """Harvested audio standing for one speaker, and where it sits in the primer.

    ``primer_t0``/``primer_t1`` are filled once the primer is assembled: they are
    what the read-back looks at, so a reference without them cannot anchor
    anything.
    """

    speaker: str
    pcm: NDArray[np.float32] = field(repr=False)
    primer_t0: float = 0.0
    primer_t1: float = 0.0

    @property
    def duration_s(self) -> float:
        """Length of the harvested audio, in the primer's time base."""
        return max(0.0, self.primer_t1 - self.primer_t0)


def plan_refresh(
    *, now_s: float, last_refresh_s: float, policy: RediarPolicy
) -> tuple[float, float] | None:
    """Decide whether a refresh is due, and over which window.

    Returns
    -------
    tuple of float, or None
        ``(window_start_s, window_end_s)`` when due, else ``None``. Clamped at
        zero, so an early refresh re-reads only the audio that exists.

    Examples
    --------
    >>> policy = RediarPolicy(window_s=180.0, period_s=60.0)
    >>> plan_refresh(now_s=30.0, last_refresh_s=0.0, policy=policy) is None
    True
    >>> plan_refresh(now_s=60.0, last_refresh_s=0.0, policy=policy)
    (0.0, 60.0)
    >>> plan_refresh(now_s=600.0, last_refresh_s=540.0, policy=policy)
    (420.0, 600.0)
    """
    if now_s - last_refresh_s < policy.period_s:
        return None
    return (max(0.0, now_s - policy.window_s), now_s)


def read_back_identities(
    segments: list[tuple[float, float, str]],
    references: list[SpeakerReference],
) -> dict[str, str]:
    """Recover global speaker identity by reading which local label covered each reference.

    The engine diarized ``[primer | window]`` in one call, so its own clustering
    already placed each window turn with the matching reference. Nothing needs to
    be matched, compared or thresholded: for each reference span, whichever local
    label covers most of it **is** that speaker.

    This is the part ported from the native block-wise diarizer, and it is why that
    design needs no embeddings of its own.

    Parameters
    ----------
    segments : list of tuple
        ``(t0, t1, local_label)`` over the primer-plus-window time base.
    references : list of SpeakerReference
        With their primer spans filled in.

    Returns
    -------
    dict of str
        ``{local_label: global_speaker}``. A local label absent from the result
        covered no reference: it is a **candidate new speaker**, and the caller
        promotes it only if it clears ``min_speaker_s`` — guessing instead would
        merge a passer-by into a participant.

    Examples
    --------
    >>> import numpy as np
    >>> silence = np.zeros(1, dtype=np.float32)
    >>> refs = [SpeakerReference("S0", silence, 0.0, 12.0),
    ...         SpeakerReference("S1", silence, 12.8, 24.8)]
    >>> segs = [(0.0, 12.0, "c1"), (12.8, 24.8, "c0"), (30.0, 40.0, "c2")]
    >>> read_back_identities(segs, refs)
    {'c1': 'S0', 'c0': 'S1'}

    ``c2`` is deliberately absent: it covered no reference, so it is a newcomer and
    not something to be guessed at.

    A reference nobody covered yields nothing for that speaker, rather than a wrong
    answer:

    >>> read_back_identities([(50.0, 60.0, "c9")], refs)
    {}
    """
    import collections

    out: dict[str, str] = {}
    for reference in references:
        if reference.duration_s <= 0:
            continue
        covered: collections.Counter = collections.Counter()
        for t0, t1, label in segments:
            overlap = min(t1, reference.primer_t1) - max(t0, reference.primer_t0)
            if overlap > 0:
                covered[label] += overlap
        if covered:
            out[covered.most_common(1)[0][0]] = reference.speaker
    return out
