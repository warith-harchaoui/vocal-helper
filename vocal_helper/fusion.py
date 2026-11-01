"""
vocal_helper.fusion
===================

Putting words and speakers back together.

Both paid modes split the two questions "what was said?" and "who said it?"
between two providers, so something has to join the answers. That is this module,
and in the live mode it is **not optional**: Gladia does not diarize live, so a
live transcript without fusion has no speakers at all.

Extracted from ``notes_helper.live_cloud.fusion`` (same author), widened to cover
the pre-recorded path as well.

Two topologies, one arithmetic
------------------------------
*Live*: the transcriber emits word-level events as it goes, the diarizer emits
speaker boundaries as it goes, and nothing is sorted or complete. Boundaries are
paired into intervals, then each **final** utterance takes the speaker it overlaps
most.

*Pre-recorded*: the transcriber returns whole utterances with its own speaker
labels, and the diarizer returns better ones. Each utterance is **relabelled** by
the diarization segment it overlaps most, which is how a hybrid run gets local
transcription timing with hosted speaker quality.

Either way the operation is the same: attribute an interval to whoever occupies it
most. Doing it by overlap rather than by midpoint matters, because an utterance
that straddles a turn change belongs to whoever held most of it, not to whoever
happened to hold its centre.

Only finals are fused
---------------------
Partial transcripts exist for display: they are revised as the model hears more.
Fusing them would put retracted words in the record. The contract is "final wins",
which is also what any downstream report already expects.

An interval still open when fusion runs
---------------------------------------
Live diarization emits a ``start`` long before the matching ``end``: the speaker is
still talking. Such a turn is treated as covering up to ``now`` (by default, the
utterance's own end), which is the streaming equivalent of the pre-recorded path's
whole-file window. Without it, the current speaker would be the one speaker who
never gets credited.

Usage example
-------------
>>> from vocal_helper.diar_cloud import DiarEvent
>>> events = [DiarEvent("start", 0.0, "S0"), DiarEvent("end", 2.0, "S0"),
...           DiarEvent("start", 2.0, "S1")]
>>> [(t.speaker, t.start, t.end) for t in build_speaker_turns(events)]
[('S0', 0.0, 2.0), ('S1', 2.0, None)]
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

from vocal_helper.asr_cloud import AsrTranscript
from vocal_helper.cloud_media import speaker_by_overlap
from vocal_helper.diar_cloud import DiarEvent
from vocal_helper.types import CloudUtterance

__all__ = [
    "SpeakerTurn",
    "build_speaker_turns",
    "fuse_live_transcripts",
    "relabel_utterances",
]


@dataclass
class SpeakerTurn:
    """One diarized interval. ``end is None`` while the speaker is still talking."""

    speaker: str
    start: float
    end: float | None = None


def build_speaker_turns(diar_events: list[DiarEvent]) -> list[SpeakerTurn]:
    """Pair ``start`` and ``end`` events into intervals.

    Two defensive rules, because a live stream is not a well-formed document.

    An ``end`` with no open ``start`` for that speaker is ignored: the protocol
    should never emit it, and a malformed or mocked stream must not crash fusion.

    A second ``start`` for a speaker whose turn is still open **closes the stale
    one first**. A real stream keeps labels stable for a whole session, so two
    consecutive ``start`` events for the same speaker mean the first one's ``end``
    was dropped — not that one person is somehow speaking twice at once.

    Parameters
    ----------
    diar_events : list of DiarEvent
        Events in arrival order. Events missing a speaker or a timestamp are
        skipped.

    Returns
    -------
    list of SpeakerTurn
        In the order the turns opened. The last turn of each speaker may still be
        open.

    Examples
    --------
    >>> from vocal_helper.diar_cloud import DiarEvent
    >>> turns = build_speaker_turns([DiarEvent("start", 1.0, "A"),
    ...                              DiarEvent("start", 3.0, "A")])
    >>> [(t.start, t.end) for t in turns]
    [(1.0, 3.0), (3.0, None)]
    >>> build_speaker_turns([DiarEvent("end", 1.0, "ghost")])
    []
    """
    open_turns: dict[str, SpeakerTurn] = {}
    turns: list[SpeakerTurn] = []
    for event in diar_events:
        if event.speaker is None or event.timestamp is None:
            continue
        if event.kind == "start":
            stale = open_turns.get(event.speaker)
            if stale is not None and stale.end is None:
                stale.end = event.timestamp
            turn = SpeakerTurn(event.speaker, event.timestamp)
            open_turns[event.speaker] = turn
            turns.append(turn)
        elif event.kind == "end":
            turn = open_turns.pop(event.speaker, None)
            if turn is not None:
                turn.end = event.timestamp
    return turns


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Seconds shared by ``[a0, a1)`` and ``[b0, b1)``, or ``0.0`` when disjoint."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def fuse_live_transcripts(
    transcripts: list[AsrTranscript],
    diar_events: list[DiarEvent],
    *,
    now: float | None = None,
    engine: str = "gladia:solaria-1+pyannoteai:live-1",
) -> list[CloudUtterance]:
    """Label each final live utterance with the speaker it overlaps most.

    Parameters
    ----------
    transcripts : list of AsrTranscript
        Everything the transcriber emitted. Partials are ignored.
    diar_events : list of DiarEvent
        Everything the diarizer emitted.
    now : float, keyword-only, optional
        The moment fusion runs, used to close still-open turns. Defaults to each
        utterance's own end.
    engine : str, keyword-only, optional
        Recorded on every utterance, so a record names what produced it.

    Returns
    -------
    list of CloudUtterance
        In transcript order. ``speaker`` is ``"?"`` when no turn overlapped at all,
        which is honest: an unknown speaker is not speaker zero.

    Examples
    --------
    >>> from vocal_helper.asr_cloud import AsrTranscript, AsrWord
    >>> from vocal_helper.diar_cloud import DiarEvent
    >>> finals = [AsrTranscript("final", [AsrWord("oui", 0.1, 0.5, 0.9)])]
    >>> events = [DiarEvent("start", 0.0, "S0"), DiarEvent("end", 1.0, "S0")]
    >>> out = fuse_live_transcripts(finals, events)
    >>> out[0]["speaker"], out[0]["text"], out[0]["word_confidence"]
    ('S0', 'oui', [0.9])
    >>> fuse_live_transcripts([AsrTranscript("partial", [])], events)
    []
    """
    turns = build_speaker_turns(diar_events)
    out: list[CloudUtterance] = []

    for transcript in transcripts:
        if transcript.kind != "final" or not transcript.words:
            continue
        t0 = transcript.words[0].time_begin
        t1 = transcript.words[-1].time_end
        open_end = now if now is not None else t1

        shared: collections.Counter = collections.Counter()
        for turn in turns:
            turn_end = turn.end if turn.end is not None else open_end
            shared[turn.speaker] += _overlap(t0, t1, turn.start, turn_end)
        speaker = shared.most_common(1)[0][0] if shared and max(shared.values()) > 0 else "?"

        utterance: CloudUtterance = {
            "t0": t0,
            "t1": t1,
            "speaker": speaker,
            "text": " ".join(w.word for w in transcript.words),
            "words": [(w.time_begin, w.time_end, w.word) for w in transcript.words],
            "language": None,
            "word_confidence": [w.confidence for w in transcript.words],
            "engine": engine,
        }
        out.append(utterance)
    return out


def relabel_utterances(
    utterances: list[CloudUtterance],
    diar_segments: list[dict[str, Any]],
    *,
    engine: str | None = None,
) -> list[CloudUtterance]:
    """Replace each utterance's speaker with the diarization segment it overlaps most.

    This is the pre-recorded fusion, and the whole point of the hybrid mode: keep
    the transcriber's words and timings, take the diarizer's speakers.

    Parameters
    ----------
    utterances : list of CloudUtterance
        From any transcriber, with whatever speakers it guessed.
    diar_segments : list of dict
        From the diarizer, each with ``start``, ``end`` and ``speaker``. Order is
        irrelevant, and segments may genuinely overlap on cross-talk.
    engine : str, keyword-only, optional
        When given, recorded on every utterance, replacing whatever was there.

    Returns
    -------
    list of CloudUtterance
        The same utterances, relabelled. An utterance that overlaps no segment
        keeps its original speaker rather than losing it: a gap in the diarization
        is not evidence about who was talking.

    Examples
    --------
    >>> utts = [{"t0": 0.0, "t1": 1.0, "speaker": "spk_0", "text": "a",
    ...          "words": [(0.0, 1.0, "a")], "language": None}]
    >>> segs = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"}]
    >>> relabel_utterances(utts, segs)[0]["speaker"]
    'SPEAKER_01'
    >>> relabel_utterances(utts, [])[0]["speaker"]
    'spk_0'
    """
    out: list[CloudUtterance] = []
    for utterance in utterances:
        relabelled: CloudUtterance = dict(utterance)  # type: ignore[assignment]
        speaker = speaker_by_overlap(diar_segments, utterance["t0"], utterance["t1"])
        if speaker is not None:
            relabelled["speaker"] = speaker
        if engine is not None:
            relabelled["engine"] = engine
        out.append(relabelled)
    return out
