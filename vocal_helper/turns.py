"""
vocal_helper.turns
==================

Turning an engine's raw output into rows a human can read and click.

Every transcriber returns something slightly wrong for display: Gladia merges
same-speaker speech into blocks that play far ahead of where the reader's eye
sits, ElevenLabs stamps a whole overlapping interjection at one instant, sentence
splitting on a lone quote mark leaves a row with no speech in it, and a diarizer
discovers a "speaker" who turns out to be a waiter saying four words. None of that
is an engine defect, and none of it is fixable upstream — it is the gap between
*what was said* and *a transcript somebody reads*.

Extracted from ``notes_helper.cloud_transcribe`` (same author), where each rule
below was added in response to a real complaint about a real recording. It belongs
here because none of it is cloud-specific: the local path produces the same shapes
and deserves the same treatment.

The five rules, and the complaint behind each
----------------------------------------------
1. **A speaker who barely speaks is probably not a speaker.** A hosted diarizer
   will happily discover a passer-by, a waiter, a crowd. Below a duration floor
   they are dropped, and the survivors are relabelled ``S0``, ``S1``, … by
   descending talk time, so the ordering means something.
2. **Consecutive same-speaker speech merges**, unless the previous row already
   ended a sentence. Without the sentence guard, a merge glues two thoughts
   together; without the merge, one sentence spreads over five rows.
3. **An oversized row is split at a real pause.** "The click lands late" is the
   complaint: a 28-second block starts playing long before the sentence the reader
   is looking at. Cuts use word timings and prefer a silence.
4. **A row with no words in it is folded away.** Sentence splitting on « or . can
   produce a row that seeks to silence, which reads as the timestamps being wrong.
5. **A zero-width row is given a span.** Overlapping speech makes some engines
   stamp an entire interjection at one instant. It still clicks to roughly the
   right place, but it can never highlight as it plays.

Order matters, and getting it wrong is silent
---------------------------------------------
Merge first, so splitting sees whole thoughts. Split next. **Then pad, then fold**
— in that order, and not the other way round.

Rules 4 and 5 contradict each other if folding runs first. Folding calls a row
with no duration an artefact and merges it away; padding exists precisely to give
a span to a row that *has* words and no duration. So when a block whose timestamps
collapsed to one instant is split by word count — the case rule 3's word-count cut
was written for — every piece comes out zero-width, folding immediately merges them
all back into one, and the split silently achieves nothing.

Padding first resolves it: a piece with real words gets its span and is no longer
an artefact, while a piece that is genuinely only punctuation still has fewer than
two alphanumerics and is still folded away. Both rules keep their meaning.

Usage example
-------------
>>> join_words([(0.0, 0.2, " C"), (0.2, 0.4, "'est"), (0.4, 0.6, " vrai")])
"C'est vrai"
>>> join_words([(0.0, 0.2, "Hello"), (0.2, 0.4, ","), (0.4, 0.6, "world")])
'Hello, world'
"""

from __future__ import annotations

import collections
import unicodedata

from vocal_helper.types import CloudUtterance

__all__ = [
    "SENTENCE_END",
    "consolidate",
    "join_words",
]

SENTENCE_END = (".", "?", "!", "…", ".»", "?»", "!»")
"""Endings that forbid a merge: the speaker finished a thought."""

# Leading punctuation that must glue to the previous token with no inserted space:
# apostrophes and hyphens (French elision and liaison, "c'est", "c'est-à-dire"),
# plus all trailing punctuation.
_NO_SPACE_BEFORE = "'’‛`-‐‑,.;:!?)]}»…%"

_Word = tuple[float, float, str]


def join_words(words: list[_Word]) -> str:
    """Rebuild a row's text from its word tokens, respecting each engine's spacing.

    Two conventions exist and they must not be mixed. Gladia and whisper encode a
    word boundary as a **leading space** on each token (``" Après"``, ``" C"``,
    then ``"'est"`` with none), so those are concatenated verbatim: inserting our
    own spaces would shatter elisions into ``"C 'est -à -dire"``. ElevenLabs tokens
    carry no leading space, so those are space-joined, with apostrophes, hyphens
    and trailing punctuation glued to the previous token.

    Which convention is in play is **detected**, not configured: a caller should
    not have to know, and a wrong setting is silently unreadable French.

    Parameters
    ----------
    words : list of tuple
        ``(t0, t1, text)`` triplets.

    Returns
    -------
    str
        The row's text, stripped.

    Examples
    --------
    >>> join_words([])
    ''
    >>> join_words([(0.0, 1.0, " c"), (1.0, 2.0, "'est-à-dire")])
    "c'est-à-dire"
    >>> join_words([(0.0, 1.0, "oui"), (1.0, 2.0, "!")])
    'oui!'
    """
    tokens = [text for _t0, _t1, text in words if text]
    if not tokens:
        return ""
    if any(token[:1].isspace() for token in tokens):
        return "".join(tokens).strip()
    out = ""
    for token in tokens:
        if out and token[:1] not in _NO_SPACE_BEFORE:
            out += " "
        out += token
    return out.strip()


def _alphanumerics(text: str) -> int:
    """How many alphanumeric characters a string holds, accents normalised."""
    return sum(c.isalnum() for c in unicodedata.normalize("NFC", text))


def _is_degenerate(turn: CloudUtterance) -> bool:
    """Whether a row carries no readable content or no duration.

    Fewer than two alphanumerics, or a non-positive span. A genuine short
    back-channel (``"Ah."``, ``"Ok."``) has two or more and a real duration, so it
    survives — which is the point: those are data, not noise.
    """
    return (
        _alphanumerics(turn.get("text", "")) < 2
        or float(turn.get("t1", 0.0)) - float(turn.get("t0", 0.0)) <= 0.0
    )


def _is_punctuation_only(turn: CloudUtterance) -> bool:
    """Whether a row holds no alphanumeric character at all."""
    return _alphanumerics(turn.get("text", "")) == 0


def _merge_same_speaker(
    utterances: list[CloudUtterance],
    *,
    merge_gap_s: float,
    max_turn_s: float,
) -> list[CloudUtterance]:
    """Merge consecutive same-speaker rows, respecting sentence ends and a cap."""
    out: list[CloudUtterance] = []
    for utterance in utterances:
        previous = out[-1] if out else None
        mergeable = (
            previous is not None
            and previous["speaker"] == utterance["speaker"]
            and utterance["t0"] - previous["t1"] < merge_gap_s
            and utterance["t1"] - previous["t0"] < max_turn_s
            # A finished sentence is a boundary the speaker themselves drew.
            and not previous["text"].rstrip().endswith(SENTENCE_END)
        )
        if mergeable and previous is not None:
            previous["t1"] = utterance["t1"]
            previous["text"] = (previous["text"].rstrip() + " " + utterance["text"].strip()).strip()
            previous["words"] = list(previous["words"]) + list(utterance["words"])
            if "word_confidence" in previous and "word_confidence" in utterance:
                previous["word_confidence"] = list(previous["word_confidence"]) + list(
                    utterance["word_confidence"]
                )
        else:
            out.append(dict(utterance))  # type: ignore[arg-type]
    return out


def _split_long(
    utterances: list[CloudUtterance],
    *,
    max_s: float,
    target_s: float,
    gap_s: float,
    max_words: int,
) -> list[CloudUtterance]:
    """Cut oversized rows at word-level pauses so each one is a tight window."""

    def sub_row(source: CloudUtterance, words: list[_Word]) -> CloudUtterance:
        t0 = words[0][0]
        t1 = max(words[-1][1], t0)
        row: CloudUtterance = {
            "t0": t0,
            "t1": t1,
            "speaker": source["speaker"],
            "text": join_words(words),
            "words": words,
            "language": source.get("language"),
        }
        for optional in ("engine",):
            if optional in source:
                row[optional] = source[optional]  # type: ignore[literal-required]
        return row

    out: list[CloudUtterance] = []
    for utterance in utterances:
        words = list(utterance.get("words") or [])
        # Too short to bother, or no word timing to cut on: pass through untouched.
        if utterance["t1"] - utterance["t0"] <= max_s or len(words) < 4:
            out.append(utterance)
            continue
        current: list[_Word] = []
        for word in words:
            if current:
                gap = word[0] - current[-1][1]
                span = current[-1][1] - current[0][0]
                # A word-count cut as well as a time cut: a block whose times
                # collapsed to one instant has no gaps to find, and would otherwise
                # never be split at all.
                if (
                    (span >= target_s and gap >= gap_s)
                    or span >= max_s
                    or len(current) >= max_words
                ):
                    out.append(sub_row(utterance, current))
                    current = []
            current.append(word)
        if current:
            out.append(sub_row(utterance, current))
    return out


def _fold_artefacts(utterances: list[CloudUtterance]) -> list[CloudUtterance]:
    """Fold a contentless row into a same-speaker neighbour, or drop it."""
    out: list[CloudUtterance] = []
    for utterance in utterances:
        degenerate = _is_degenerate(utterance)
        if degenerate and out and out[-1]["speaker"] == utterance["speaker"]:
            previous = out[-1]
            extra = utterance.get("text", "").strip()
            if extra:
                previous["text"] = (previous["text"].rstrip() + " " + extra).strip()
            previous["t1"] = max(previous["t1"], utterance.get("t1", previous["t1"]))
            previous["words"] = list(previous["words"]) + list(utterance.get("words") or [])
        elif degenerate and _is_punctuation_only(utterance):
            continue  # lone punctuation, no same-speaker neighbour: nothing is lost
        else:
            out.append(utterance)

    # A leading artefact has no previous neighbour, so it folds FORWARD instead.
    if len(out) >= 2 and _is_degenerate(out[0]) and out[1]["speaker"] == out[0]["speaker"]:
        head, following = out[0], out[1]
        following["t0"] = head["t0"]
        following["text"] = (
            head.get("text", "").strip() + " " + following["text"].lstrip()
        ).strip()
        following["words"] = list(head.get("words") or []) + list(following["words"])
        out = out[1:]
    return out


def _pad_collapsed(utterances: list[CloudUtterance], *, rate_s: float) -> list[CloudUtterance]:
    """Give zero-width rows a span, and spread their collapsed word times."""
    for index, utterance in enumerate(utterances):
        t0, t1 = utterance["t0"], utterance["t1"]
        if t1 > t0:
            continue
        words = list(utterance.get("words") or [])
        span = rate_s * max(1, len(words))
        # Never run into the next row: a padded span that overlaps its neighbour
        # would make two rows highlight at once, which reads as a worse bug than the
        # one being fixed.
        following = utterances[index + 1]["t0"] if index + 1 < len(utterances) else t0 + span
        t1 = min(t0 + span, max(t0 + 0.05, following - 0.01))
        utterance["t1"] = t1
        count = len(words)
        if count:
            utterance["words"] = [
                (t0 + (t1 - t0) * k / count, t0 + (t1 - t0) * (k + 1) / count, text)
                for k, (_w0, _w1, text) in enumerate(words)
            ]
    return utterances


def consolidate(
    utterances: list[CloudUtterance],
    *,
    min_speaker_s: float = 120.0,
    merge_gap_s: float = 3.0,
    max_turn_s: float = 30.0,
    max_s: float = 9.0,
    target_s: float = 6.0,
    gap_s: float = 0.3,
    max_words: int = 45,
    pad_rate_s: float = 0.34,
) -> tuple[list[CloudUtterance], dict[str, str]]:
    """Turn raw engine output into readable, clickable rows with stable speaker ids.

    Applies the five rules of the module docstring, in the order that docstring
    explains, and relabels the surviving speakers ``S0``, ``S1``, … by descending
    talk time.

    Parameters
    ----------
    utterances : list of CloudUtterance
        Raw rows from any engine, with whatever speaker labels it used.
    min_speaker_s : float, keyword-only, optional
        A discovered speaker must exceed this total duration to survive. Two
        minutes by default. **Lower it** when a real but quiet participant matters;
        the default is tuned for a panel where a four-word voice is a passer-by.
    merge_gap_s, max_turn_s : float, keyword-only, optional
        Same-speaker rows closer than the gap merge, up to the cap.
    max_s, target_s, gap_s, max_words : float and int, keyword-only, optional
        When a row is cut, and where: past ``max_s`` it is split, preferring a
        silence of ``gap_s`` once the running row passes ``target_s``, and always
        before ``max_words`` tokens.
    pad_rate_s : float, keyword-only, optional
        Seconds per word given to a zero-width row. The default is a calm French
        rate, roughly three words a second.

    Returns
    -------
    tuple
        ``(rows, speakers)`` where ``speakers`` maps each new id to the original
        engine label it came from — kept so a mis-assignment can be traced back to
        what the engine actually said, rather than disappearing into a renaming.

    Examples
    --------
    >>> raw = [
    ...     {"t0": 0.0, "t1": 5.0, "speaker": "spk_1", "text": "Bonjour à tous",
    ...      "words": [(0.0, 5.0, "Bonjour à tous")], "language": "fr"},
    ...     {"t0": 5.0, "t1": 6.0, "speaker": "waiter", "text": "Un café",
    ...      "words": [(5.0, 6.0, "Un café")], "language": "fr"},
    ... ]
    >>> rows, speakers = consolidate(raw, min_speaker_s=2.0)
    >>> [(r["speaker"], r["text"]) for r in rows]
    [('S0', 'Bonjour à tous')]
    >>> speakers
    {'S0': 'spk_1'}

    With no floor, the brief voice survives and is simply ranked second:

    >>> rows, speakers = consolidate(raw, min_speaker_s=0.0)
    >>> [r["speaker"] for r in rows], speakers
    (['S0', 'S1'], {'S0': 'spk_1', 'S1': 'waiter'})
    """
    talk_time: collections.Counter = collections.Counter()
    for utterance in utterances:
        speaker = utterance.get("speaker")
        if speaker:
            talk_time[speaker] += max(0.0, utterance["t1"] - utterance["t0"])

    survivors = [spk for spk, total in talk_time.most_common() if total >= min_speaker_s]
    renamed = {spk: f"S{index}" for index, spk in enumerate(survivors)}

    kept: list[CloudUtterance] = []
    for utterance in utterances:
        speaker = utterance.get("speaker")
        if speaker not in renamed:
            continue  # below the floor, or never attributed at all
        row: CloudUtterance = dict(utterance)  # type: ignore[assignment]
        row["speaker"] = renamed[speaker]
        kept.append(row)

    rows = _merge_same_speaker(kept, merge_gap_s=merge_gap_s, max_turn_s=max_turn_s)
    rows = _split_long(rows, max_s=max_s, target_s=target_s, gap_s=gap_s, max_words=max_words)
    # Pad BEFORE folding, or the two rules cancel each other out — see the module
    # docstring. This ordering is the fix for a real defect inherited from the code
    # this module was extracted from.
    rows = _pad_collapsed(rows, rate_s=pad_rate_s)
    rows = _fold_artefacts(rows)

    return rows, {new: old for old, new in renamed.items()}
