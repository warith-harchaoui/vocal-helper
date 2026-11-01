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

DEFAULT_MIN_SPEAKER_S = 5.0
"""In-window speech before a new cluster becomes a speaker.

**Not the native block-wise diarizer's 15 s, and the difference matters.** That value
was tuned for windows cut from a whole recording, where a real participant holds the
floor a good stretch within *some* window. Here the window is bounded and short, and a
genuine but quiet participant can legitimately hold it for ten seconds in three
minutes — a 15 s floor then never promotes them, the speaker count never grows past
one, and the refresh degenerates into agreeing with whatever it already believed.
Measured on AMI: that is exactly what happened.

Five seconds still clears a spurious fragment, which lasts under two.
"""

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
    max_speakers: int = 12
    """Hard ceiling on discovered speakers.

    The lesson of ``DIARIZATION-TROUBLES.md``: an unbounded greedy clusterer minted
    **203 labels for a four-speaker podcast**. A ceiling is not a quality knob, it
    is the difference between a transcript and a list of 203 strangers. Twelve is
    past any panel and short of any runaway.
    """

    stable_rounds_before_pinning: int = 1
    """Consecutive refreshes with no new speaker before the count gets pinned.

    Discovery and stability pull opposite ways, and both are needed. While the count
    is still growing, the diarizer must be free to find a voice nobody has heard yet,
    so it runs with a **floor**. Once the set has stopped growing, the count is
    **pinned**, and that is what makes cross-window identity clean: the engine
    returns exactly as many clusters as there are references, every one of them maps
    back to a reference, and there is nothing left over to mint a new name from.

    Measured: without this, the identity drifted *worse than the online stage it was
    meant to fix* — seven labels for four speakers, long-range agreement 0.297
    against the online stage's 0.969.
    """

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
    *,
    min_coverage: float = 0.0,
) -> dict[str, str]:
    """Recover global speaker identity by reading which local label covered each reference.

    The engine diarized ``[primer | window]`` in one call, so its own clustering
    already placed each window turn with the matching reference. Identity therefore
    travels by **time**, not by embedding geometry: for each reference span, read
    which local label covered it.

    This is the part ported from the native block-wise diarizer, and it is why that
    design needs no embeddings of its own.

    The read-back is a **bijection**, so it is resolved by optimal assignment and
    not by a per-reference argmax. The distinction is not cosmetic. A greedy
    argmax lets one local label win two references — which happens whenever the
    engine merges two primer speakers into a single cluster — and the second
    write then **overwrites** the first, so one known speaker silently loses its
    label and is reported as absent. Optimal assignment cannot do that: the
    merged label takes one reference, the other reference stays unmatched, and
    the caller sees a speaker it could not place instead of a wrong answer.

    Parameters
    ----------
    segments : list of tuple
        ``(t0, t1, local_label)`` over the primer-plus-window time base.
    references : list of SpeakerReference
        With their primer spans filled in.
    min_coverage : float
        Fraction of a reference's own duration that the matched label must cover
        for the pairing to count. ``0.0`` (default) accepts any shared time at
        all; raising it refuses a label that merely brushed the reference.

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
    {'c0': 'S1', 'c1': 'S0'}

    ``c2`` is deliberately absent: it covered no reference, so it is a newcomer and
    not something to be guessed at.

    A reference nobody covered yields nothing for that speaker, rather than a wrong
    answer:

    >>> read_back_identities([(50.0, 60.0, "c9")], refs)
    {}

    One label straddling **both** references — the engine merged two primer
    speakers — is assigned to one of them, and the other is reported as unplaced.
    A per-reference argmax would instead have overwritten the first mapping and
    lost ``S0`` without a word:

    >>> read_back_identities([(0.0, 24.8, "c0")], refs)
    {'c0': 'S0'}
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    usable = [r for r in references if r.duration_s > 0]
    if not usable:
        return {}
    labels = sorted({label for _, _, label in segments})
    if not labels:
        return {}

    shared = np.zeros((len(labels), len(usable)), dtype=np.float64)
    index = {label: i for i, label in enumerate(labels)}
    for t0, t1, label in segments:
        i = index[label]
        for j, reference in enumerate(usable):
            overlap = min(t1, reference.primer_t1) - max(t0, reference.primer_t0)
            if overlap > 0:
                shared[i, j] += overlap

    out: dict[str, str] = {}
    rows, cols = linear_sum_assignment(-shared)
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        if shared[i, j] <= 0:
            continue  # that pairing rests on no shared time at all
        if shared[i, j] < min_coverage * usable[j].duration_s:
            continue  # it brushed the reference rather than covering it
        out[labels[i]] = usable[j].speaker
    return out


# --------------------------------------------------------------------------- #
#  Assembling the primer, and reconstructing the window
# --------------------------------------------------------------------------- #


def harvest_room_tone(
    pcm: NDArray[np.float32], sample_rate: int, gap_s: float
) -> NDArray[np.float32]:
    """The quietest contiguous ``gap_s`` of this recording, to splice between pieces.

    Ported from the native block-wise diarizer. The recording's **own** quietest
    ambience is both the right level and the right *nature* of silence: a
    segmentation model trained on real audio has never seen digital zeros, and
    behaves oddly around them. Spliced zeros are not a neutral separator, they are
    an out-of-distribution input placed exactly where a clean cut is needed.

    Found by prefix sums of squares, so the scan is linear in the buffer rather
    than quadratic in the number of candidate windows.

    Parameters
    ----------
    pcm : NDArray[np.float32]
        Mono audio to harvest from.
    sample_rate : int
        Its sample rate.
    gap_s : float
        Length to harvest.

    Returns
    -------
    NDArray[np.float32]
        The harvested span. True zeros **only** when the buffer is shorter than one
        gap, which is a degenerate case, not a choice.

    Examples
    --------
    >>> sr = 16000
    >>> loud = np.ones(sr, dtype=np.float32)
    >>> quiet = np.full(sr, 0.001, dtype=np.float32)
    >>> room = harvest_room_tone(np.concatenate([loud, quiet]), sr, 0.5)
    >>> len(room) == sr // 2 and float(np.abs(room).max()) < 0.01
    True
    >>> len(harvest_room_tone(np.zeros(10, dtype=np.float32), sr, 0.5))
    8000
    """
    window = int(gap_s * sample_rate)
    if window <= 0:
        return np.zeros(1, dtype=np.float32)
    if pcm.size < window:
        return np.zeros(window, dtype=np.float32)

    squares = np.concatenate([[0.0], np.cumsum(np.asarray(pcm, dtype=np.float64) ** 2)])
    hop = max(1, sample_rate // 100)
    starts = np.arange(0, pcm.size - window + 1, hop)
    energies = squares[starts + window] - squares[starts]
    best = int(starts[int(np.argmin(energies))])
    return np.asarray(pcm[best : best + window], dtype=np.float32)


def build_references(
    *,
    pcm: NDArray[np.float32],
    sample_rate: int,
    segments: list[tuple[float, float, str]],
    offset_s: float,
    budget_s: float,
) -> dict[str, NDArray[np.float32]]:
    """Harvest up to ``budget_s`` of each speaker's own audio, longest turns first.

    Longest first, because a long turn is the one most likely to be a single voice
    speaking alone: short turns are where overlap and mislabelling live, and a
    reference built from them anchors the wrong person.

    Parameters
    ----------
    pcm : NDArray[np.float32], keyword-only
        The window's audio.
    sample_rate : int, keyword-only
    segments : list of tuple, keyword-only
        ``(t0, t1, speaker)`` in absolute time.
    offset_s : float, keyword-only
        Absolute time of ``pcm[0]``, so segment times can be placed in it.
    budget_s : float, keyword-only
        Per-speaker audio to harvest.

    Returns
    -------
    dict of ndarray
        One concatenated clip per speaker. A speaker whose audio falls outside the
        window is absent rather than represented by an empty clip.

    Examples
    --------
    >>> sr = 1000
    >>> pcm = np.arange(5000, dtype=np.float32)
    >>> segs = [(0.0, 1.0, "A"), (1.0, 4.0, "B"), (4.0, 5.0, "A")]
    >>> refs = build_references(pcm=pcm, sample_rate=sr, segments=segs,
    ...                         offset_s=0.0, budget_s=2.0)
    >>> sorted(refs), [len(refs[k]) for k in sorted(refs)]
    (['A', 'B'], [2000, 2000])
    """
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for t0, t1, speaker in segments:
        by_speaker.setdefault(speaker, []).append((t0, t1))

    out: dict[str, NDArray[np.float32]] = {}
    budget_samples = int(budget_s * sample_rate)
    for speaker, spans in by_speaker.items():
        pieces: list[NDArray[np.float32]] = []
        taken = 0
        for t0, t1 in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
            lo = max(0, int((t0 - offset_s) * sample_rate))
            hi = min(pcm.size, int((t1 - offset_s) * sample_rate))
            if hi <= lo:
                continue
            piece = pcm[lo : lo + min(hi - lo, budget_samples - taken)]
            if piece.size == 0:
                continue
            pieces.append(np.asarray(piece, dtype=np.float32))
            taken += piece.size
            if taken >= budget_samples:
                break
        if pieces:
            out[speaker] = np.concatenate(pieces)
    return out


def assemble_primer(
    references: dict[str, NDArray[np.float32]],
    room: NDArray[np.float32],
    sample_rate: int,
) -> tuple[NDArray[np.float32], list[SpeakerReference]]:
    """Splice the references into one primer, separated by room tone.

    A gap goes **before** each reference and once more at the end, so the
    segmentation model cuts a boundary between every pair of references *and*
    between the last reference and the window that follows. Without that final gap
    the last reference merges into the window and anchors nothing.

    Parameters
    ----------
    references : dict of ndarray
        Per-speaker clips, from :func:`build_references`.
    room : ndarray
        One gap of harvested room tone, from :func:`harvest_room_tone`.
    sample_rate : int

    Returns
    -------
    tuple
        ``(primer_pcm, references)`` where each :class:`SpeakerReference` carries
        the span it occupies in the primer's own time base — which is what
        :func:`read_back_identities` reads.

    Examples
    --------
    >>> sr = 1000
    >>> room = np.zeros(800, dtype=np.float32)
    >>> primer, refs = assemble_primer(
    ...     {"A": np.ones(1000, dtype=np.float32), "B": np.ones(2000, dtype=np.float32)},
    ...     room, sr)
    >>> len(primer) == 800 + 1000 + 800 + 2000 + 800
    True
    >>> [(r.speaker, round(r.primer_t0, 1), round(r.primer_t1, 1)) for r in refs]
    [('A', 0.8, 1.8), ('B', 2.6, 4.6)]
    """
    chunks: list[NDArray[np.float32]] = []
    spans: list[SpeakerReference] = []
    cursor = 0
    for speaker in sorted(references):
        clip = references[speaker]
        chunks.append(room)
        cursor += room.size
        start = cursor
        chunks.append(clip)
        cursor += clip.size
        spans.append(SpeakerReference(speaker, clip, start / sample_rate, cursor / sample_rate))
    if chunks:
        chunks.append(room)
    primer = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return primer, spans


def reconstruct_window(
    voiced: list[tuple[float, float, NDArray[np.float32]]],
    room: NDArray[np.float32],
    sample_rate: int,
    *,
    window_start_s: float,
) -> NDArray[np.float32]:
    """Rebuild a contiguous window from voiced spans, filling gaps with room tone.

    The offline diarizer wants continuous audio, and what arrives from
    voice-activity detection is speech with the silence removed. Concatenating the
    voiced spans would compress the timeline and every returned boundary would be
    wrong; padding with digital zeros would place out-of-distribution input in
    every gap. So the gaps are filled with the recording's own ambience, tiled.

    Parameters
    ----------
    voiced : list of tuple
        ``(t0, t1, pcm)`` in absolute time, in order.
    room : ndarray
        Harvested room tone, tiled to fill each gap.
    sample_rate : int
    window_start_s : float, keyword-only
        Absolute time the reconstructed window begins at.

    Returns
    -------
    ndarray
        Audio whose sample ``i`` is at absolute time
        ``window_start_s + i / sample_rate``.

    Examples
    --------
    >>> sr = 1000
    >>> room = np.full(100, 0.001, dtype=np.float32)
    >>> speech = np.ones(500, dtype=np.float32)
    >>> out = reconstruct_window([(1.0, 1.5, speech), (2.0, 2.5, speech)],
    ...                          room, sr, window_start_s=1.0)
    >>> len(out)  # 0.5 s speech + 0.5 s gap + 0.5 s speech
    1500
    >>> round(float(out[700]), 4)  # inside the gap: ambience, not zeros
    0.001
    """
    if not voiced:
        return np.zeros(0, dtype=np.float32)

    total = int(round((voiced[-1][1] - window_start_s) * sample_rate))
    if room.size:
        canvas = np.tile(room, total // room.size + 1)[:total].astype(np.float32).copy()
    else:
        canvas = np.zeros(total, dtype=np.float32)
    for t0, _t1, pcm in voiced:
        lo = int(round((t0 - window_start_s) * sample_rate))
        if lo < 0:
            pcm = pcm[-lo:]
            lo = 0
        hi = min(total, lo + pcm.size)
        if hi > lo:
            canvas[lo:hi] = np.asarray(pcm[: hi - lo], dtype=np.float32)
    return canvas


# --------------------------------------------------------------------------- #
#  The stage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LabelRevision:
    """One already-published span whose speaker turned out to be somebody else."""

    t0: float
    t1: float
    old_speaker: str
    new_speaker: str


class PeriodicOfflineDiarStage:
    """Live diarization at offline quality: a fast provisional label, then a correction.

    Composes rather than replaces. The existing online stage gives a label within
    about half a second, which is what a live display needs; this stage then runs
    the **offline** diarizer periodically over a bounded window and reports the
    labels that were wrong. Neither half is sufficient alone: the online stage is
    three to four times worse, and the offline one cannot answer in half a second.

    The audio path stays once-only: a revision carries **labels**, never audio, so
    nothing downstream re-transcribes anything. Transcription hears each segment
    exactly as many times as the online stage would have let it.

    Parameters
    ----------
    online : Any
        An :class:`~vocal_helper.diar.OnlineDiarStage`-shaped object, used for the
        provisional label. Injected so this class is testable without an embedder.
    offline : Any
        An :class:`~vocal_helper.diar.OfflineDiarStage`-shaped object exposing
        ``diarize(pcm, sr, *, num_speakers=...)``. Injected for the same reason.
    policy : RediarPolicy, optional
    on_revision : callable, optional
        ``(list[LabelRevision]) -> None``, called after each refresh. ``None``
        means revisions are only recorded on :attr:`revisions`, which is the right
        default for a batch caller and the wrong one for a live display — a display
        that ignores revisions shows labels it has been told are wrong.

    Notes
    -----
    **Labels change retroactively, and that has to be visible.** A line shown as
    ``S1`` may become ``S2``. Keeping a label known to be wrong would be worse, and
    silently rewriting history would be worse still: the display must re-colour,
    which is a deliberate behaviour, not an accident of ordering.
    """

    def __init__(
        self,
        *,
        online: object,
        offline: object,
        policy: RediarPolicy | None = None,
        on_revision: object | None = None,
    ) -> None:
        self.online = online
        self.offline = offline
        self.policy = policy or RediarPolicy()
        self.on_revision = on_revision
        #: Every revision this stage has produced, newest last.
        self.revisions: list[LabelRevision] = []
        #: Refresh count, for a caller that wants to report the cadence.
        self.refreshes = 0

        self._published: list[dict] = []
        self._voiced: list[tuple[float, float, NDArray[np.float32]]] = []
        self._references: dict[str, NDArray[np.float32]] = {}
        self._room: NDArray[np.float32] | None = None
        self._last_refresh_s = 0.0
        #: Consecutive refreshes that discovered nobody new. Drives the switch from
        #: discovery (a floor) to pinned (an exact count) — see the policy.
        self._stable_rounds = 0
        # Learned from the first frame; the fallback only matters for a stream that
        # closes without ever delivering one.
        self._sample_rate = 16000

    # ----- what was discovered -------------------------------------------

    @property
    def discovered_speakers(self) -> int:
        """How many speakers have been **discovered**, never assumed.

        Determining the number of people talking is part of what diarization does,
        not an input to it. Nothing in the free path forces a count: the first
        refresh runs the engine in auto-detect, each later one runs with a floor
        while the set is still growing, and only once it has stopped growing is the
        count pinned — to the number discovered, not to a number supplied.

        A caller that genuinely knows the count can still pass hints, and a caller
        that does not gets an answer rather than a guess.

        Returns
        -------
        int
            Zero before the first refresh.
        """
        return len(self._references)

    @property
    def speaker_labels(self) -> list[str]:
        """The discovered speakers, in discovery order."""
        return sorted(self._references)

    # ----- the refresh ----------------------------------------------------

    def refresh(
        self, *, now_s: float, sample_rate: int, force: bool = False
    ) -> list[LabelRevision]:
        """Re-diarize the trailing window and return the labels that were wrong.

        Returns an empty list when no refresh is due, when there is nothing in the
        window, or when the offline diarizer declined — a refresh that fails must
        cost the provisional labels nothing.

        Parameters
        ----------
        now_s : float, keyword-only
            Audio time reached.
        sample_rate : int, keyword-only
        force : bool, keyword-only, optional
            Refresh even if the period has not elapsed. Used when the stream closes:
            the period is a rate limit for a session that continues, and this one does
            not. Resetting the clock instead does not work — the period is measured
            forward from the last refresh, not backward from now.

        Returns
        -------
        list of LabelRevision
        """
        if force:
            plan: tuple[float, float] | None = (max(0.0, now_s - self.policy.window_s), now_s)
        else:
            plan = plan_refresh(
                now_s=now_s, last_refresh_s=self._last_refresh_s, policy=self.policy
            )
        if plan is None:
            return []
        window_start, window_end = plan
        self._last_refresh_s = now_s

        voiced = [v for v in self._voiced if v[1] > window_start]
        if not voiced:
            return []

        if self._room is None or self._room.size == 0:
            # Harvest once, from the quietest span of everything heard so far: the
            # ambience of a room does not change, and re-harvesting every refresh
            # would spend the scan again for the same answer.
            everything = np.concatenate([v[2] for v in self._voiced])
            self._room = harvest_room_tone(everything, sample_rate, self.policy.gap_s)

        window = reconstruct_window(voiced, self._room, sample_rate, window_start_s=window_start)
        primer, references = assemble_primer(self._references, self._room, sample_rate)
        audio = np.concatenate([primer, window]) if primer.size else window
        shift = primer.size / sample_rate

        # DISCOVER, THEN PIN — the discipline ported from the native block-wise
        # diarizer, and the half I had left out.
        #
        # While the speaker set is still growing, the engine runs with a **floor**:
        # the references are physically in the primer so at least that many voices
        # are present, and a voice nobody has heard yet must stay findable. Passing
        # the count as an exact ``num_speakers`` here is self-fulfilling — with one
        # reference every later refresh is forced to find one speaker, so a second
        # never appears, and the error rate stays *exactly* the online stage's.
        #
        # Once the set stops growing, the count is **pinned**, and that is what makes
        # cross-window identity clean: the engine returns exactly as many clusters as
        # there are references, every one maps back through the read-back, and nothing
        # is left over to mint a new name from. Without the pin, identity drifted
        # worse than the online stage it was meant to repair.
        known = len(self._references)
        pinned = bool(known) and self._stable_rounds >= self.policy.stable_rounds_before_pinning
        hints: dict = {}
        if pinned:
            hints["num_speakers"] = known
        elif known:
            hints["min_speakers"] = known
        try:
            segments = self.offline.diarize(audio, sample_rate, **hints)
        except Exception:  # noqa: BLE001 - a failed refresh must not lose the labels we have
            import os_helper as osh

            osh.warning(
                "  rediar: refresh failed; keeping the provisional labels. The next "
                "refresh will try again."
            )
            return []

        self.refreshes += 1
        mapping = read_back_identities(segments, references)
        before = len(self._references)

        # Window turns, back in absolute time, with the primer's offset removed.
        absolute: list[tuple[float, float, str]] = [
            (t0 - shift + window_start, t1 - shift + window_start, label)
            for t0, t1, label in segments
            if t0 >= shift
        ]

        # A local label nobody's reference covered is a candidate newcomer, and it
        # is promoted only if it speaks past the floor — guessing would merge a
        # passer-by into a participant, the one error reading cannot repair.
        unknown: dict[str, float] = {}
        for t0, t1, label in absolute:
            if label not in mapping:
                unknown[label] = unknown.get(label, 0.0) + max(0.0, t1 - t0)
        promoted = 0
        next_index = len(self._references)
        if not pinned:
            for label, spoken in sorted(unknown.items(), key=lambda kv: -kv[1]):
                if next_index >= self.policy.max_speakers:
                    import os_helper as osh

                    osh.warning(
                        f"  rediar: speaker ceiling of {self.policy.max_speakers} reached; "
                        f"further clusters stay unlabelled rather than inflating the count"
                    )
                    break
                if spoken >= self.policy.min_speaker_s:
                    mapping[label] = f"S{next_index}"
                    next_index += 1
                    promoted += 1
        # Stability is what flips discovery into pinning. A refresh that found nobody
        # new is evidence the set is complete; one that found somebody resets the
        # count, because a newcomer means the room changed.
        self._stable_rounds = self._stable_rounds + 1 if promoted == 0 else 0
        if promoted:
            import os_helper as osh

            osh.info(
                f"  rediar: discovered {promoted} new speaker(s) "
                f"({before} → {before + promoted}); the count is found, not supplied"
            )

        revisions = self._apply(absolute, mapping, window_start=window_start)
        self._harvest_references(window, sample_rate, absolute, mapping, offset_s=window_start)
        if revisions and callable(self.on_revision):
            self.on_revision(revisions)
        self.revisions.extend(revisions)
        return revisions

    def _apply(
        self,
        absolute: list[tuple[float, float, str]],
        mapping: dict[str, str],
        *,
        window_start: float,
    ) -> list[LabelRevision]:
        """Compare the refresh against what was published, and record the differences."""
        import collections

        revisions: list[LabelRevision] = []
        for row in self._published:
            if row["t1"] <= window_start:
                continue  # outside the window; this refresh says nothing about it
            overlap: collections.Counter = collections.Counter()
            for t0, t1, label in absolute:
                shared = min(row["t1"], t1) - max(row["t0"], t0)
                if shared > 0 and label in mapping:
                    overlap[mapping[label]] += shared
            if not overlap:
                continue
            best = overlap.most_common(1)[0][0]
            if best != row["speaker"]:
                revisions.append(LabelRevision(row["t0"], row["t1"], row["speaker"], best))
                row["speaker"] = best
        return revisions

    def _harvest_references(
        self,
        window: NDArray[np.float32],
        sample_rate: int,
        absolute: list[tuple[float, float, str]],
        mapping: dict[str, str],
        *,
        offset_s: float,
    ) -> None:
        """Keep one reference clip per known speaker, refreshed from this window."""
        named = [(t0, t1, mapping[label]) for t0, t1, label in absolute if label in mapping]
        harvested = build_references(
            pcm=window,
            sample_rate=sample_rate,
            segments=named,
            offset_s=offset_s,
            budget_s=self.policy.ref_budget_s,
        )
        # Replace rather than accumulate: a reference from the recent past tracks a
        # voice that changed (a head turn, a different microphone distance) better
        # than the first clip ever harvested. A speaker absent from this window
        # keeps the reference they had — losing it would make them a newcomer again.
        self._references.update(harvested)

    # ----- the stage contract --------------------------------------------

    async def run(self, inbox, outbox) -> None:  # noqa: ANN001 - asyncio.Queue, duck-typed
        """Consume voiced segments, publish provisionally, and revise periodically.

        Same contract as the online stage, so this is a drop-in: a consumer that
        reads ``outbox`` needs no change. What is new is that labels already in
        ``outbox`` may be superseded, which is what ``on_revision`` reports.
        """
        while True:
            item = await inbox.get()
            if item is None:
                # A final refresh before closing: the last window has never been
                # re-diarized, and it is the one the reader is still looking at.
                if self._voiced:
                    # The period is a rate limit for a session that continues; this one
                    # does not, and the last window is what the reader is still looking at.
                    self.refresh(
                        now_s=self._voiced[-1][1],
                        sample_rate=self._sample_rate,
                        force=True,
                    )
                await outbox.put(None)
                return

            self._sample_rate = int(item["sample_rate"])
            self._voiced.append((item["t0"], item["t1"], item["pcm"]))
            # Keep only what a window plus a period could ever need; the rest is
            # memory that will never be re-read, and a live session is unbounded.
            horizon = item["t1"] - (self.policy.window_s + self.policy.period_s)
            self._voiced = [v for v in self._voiced if v[1] > horizon]

            labelled = self.online._label(item)
            if labelled is not None:
                self._published.append(
                    {
                        "t0": labelled["t0"],
                        "t1": labelled["t1"],
                        "speaker": labelled["speaker"],
                    }
                )
                await outbox.put(labelled)

            self.refresh(now_s=item["t1"], sample_rate=self._sample_rate)
