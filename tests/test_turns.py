"""Turning raw engine output into rows a human reads and clicks.

Each test below names the complaint it prevents, because that is the only reason
any of these rules exist: none of them is a correctness fix on the engines, all of
them are the gap between *what was said* and *a transcript somebody reads*.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

from vocal_helper.turns import consolidate, join_words


def _row(t0, t1, speaker, text, words=None, language="fr"):
    """One raw row, with word timings spread evenly unless given."""
    if words is None:
        pieces = text.split()
        step = (t1 - t0) / max(1, len(pieces))
        words = [(t0 + i * step, t0 + (i + 1) * step, w) for i, w in enumerate(pieces)]
    return {
        "t0": t0,
        "t1": t1,
        "speaker": speaker,
        "text": text,
        "words": words,
        "language": language,
    }


# --------------------------------------------------------------------------- #
#  Rebuilding text from tokens
# --------------------------------------------------------------------------- #


def test_elisions_survive_the_leading_space_convention():
    """Inserting our own spaces would shatter French elision into "C 'est -à -dire",
    which is the single most visible way to make a transcript look machine-made."""
    tokens = [(0.0, 0.1, " C"), (0.1, 0.2, "'est"), (0.2, 0.3, "-à"), (0.3, 0.4, "-dire")]
    assert join_words(tokens) == "C'est-à-dire"


def test_tokens_without_leading_spaces_are_space_joined_but_punctuation_glues():
    """The other convention. Getting it backwards gives either "Helloworld" or
    "Hello , world", and a caller should not have to know which engine it has."""
    tokens = [(0.0, 0.1, "Bonjour"), (0.1, 0.2, ","), (0.2, 0.3, "Marie"), (0.3, 0.4, "!")]
    assert join_words(tokens) == "Bonjour, Marie!"


def test_no_words_gives_no_text_rather_than_an_error():
    assert join_words([]) == ""


# --------------------------------------------------------------------------- #
#  Rule 1 — a speaker who barely speaks is probably not a speaker
# --------------------------------------------------------------------------- #


def test_a_four_word_voice_is_dropped_and_the_rest_ranked_by_talk_time():
    """A hosted diarizer discovers passers-by. Keeping them puts a waiter in the
    speaking-time table of a political panel."""
    raw = [
        _row(0.0, 8.0, "spk_2", "une longue intervention de fond"),
        _row(8.0, 9.0, "passerby", "un café"),
        _row(9.0, 40.0, "spk_5", "une intervention beaucoup plus longue encore"),
    ]
    rows, speakers = consolidate(raw, min_speaker_s=5.0)
    assert speakers == {"S0": "spk_5", "S1": "spk_2"}  # by descending talk time
    assert {r["speaker"] for r in rows} == {"S0", "S1"}


def test_the_mapping_back_to_the_engine_label_is_kept():
    """A renaming that lost the original makes a mis-assignment untraceable: there
    would be no way to ask what the engine actually said."""
    rows, speakers = consolidate([_row(0.0, 10.0, "SPEAKER_07", "bonjour")], min_speaker_s=1.0)
    assert speakers == {"S0": "SPEAKER_07"}
    assert rows[0]["speaker"] == "S0"


def test_an_unattributed_row_is_dropped_rather_than_invented():
    """A row the diarizer never attributed has no speaker. Assigning it one would
    put words in somebody's mouth."""
    raw = [_row(0.0, 10.0, "a", "attribué"), _row(10.0, 20.0, None, "orphelin")]
    rows, _ = consolidate(raw, min_speaker_s=1.0)
    assert [r["text"] for r in rows] == ["attribué"]


# --------------------------------------------------------------------------- #
#  Rule 2 — merging, and the sentence that forbids it
# --------------------------------------------------------------------------- #


def test_consecutive_same_speaker_speech_merges():
    """Without this, one sentence spreads over five clickable rows."""
    raw = [_row(0.0, 2.0, "a", "je pense que"), _row(2.2, 4.0, "a", "nous devrions")]
    rows, _ = consolidate(raw, min_speaker_s=0.0, max_s=100.0)
    assert len(rows) == 1
    assert rows[0]["text"] == "je pense que nous devrions"
    assert (rows[0]["t0"], rows[0]["t1"]) == (0.0, 4.0)


def test_a_finished_sentence_is_a_boundary_the_speaker_drew():
    """Merging across a full stop glues two thoughts together, which changes what
    the transcript says the person meant."""
    raw = [_row(0.0, 2.0, "a", "c'est terminé."), _row(2.2, 4.0, "a", "autre chose")]
    rows, _ = consolidate(raw, min_speaker_s=0.0, max_s=100.0)
    assert [r["text"] for r in rows] == ["c'est terminé.", "autre chose"]


def test_a_different_speaker_never_merges():
    raw = [_row(0.0, 2.0, "a", "je pense"), _row(2.1, 4.0, "b", "moi non")]
    rows, _ = consolidate(raw, min_speaker_s=0.0, max_s=100.0)
    assert len(rows) == 2


def test_a_long_silence_breaks_the_merge():
    raw = [_row(0.0, 2.0, "a", "je pense"), _row(30.0, 32.0, "a", "finalement")]
    rows, _ = consolidate(raw, min_speaker_s=0.0, merge_gap_s=3.0, max_s=100.0)
    assert len(rows) == 2


# --------------------------------------------------------------------------- #
#  Rule 3 — "the click lands late"
# --------------------------------------------------------------------------- #


def test_an_oversized_row_is_cut_at_a_silence():
    """A 28-second block starts playing long before the sentence the reader is
    looking at. Cutting at a real pause keeps each row a tight window."""
    words = [(float(i), float(i) + 0.5, f"mot{i}") for i in range(20)]
    words[10] = (10.0, 10.4, "mot10")  # a gap before this one
    raw = [
        {
            "t0": 0.0,
            "t1": 20.0,
            "speaker": "a",
            "text": " ".join(w[2] for w in words),
            "words": words,
            "language": "fr",
        }
    ]
    rows, _ = consolidate(raw, min_speaker_s=0.0, max_s=9.0, target_s=6.0, gap_s=0.3)
    assert len(rows) > 1
    assert all(r["t1"] - r["t0"] <= 12.0 for r in rows)
    # Nothing is lost by the cut: every word is still somewhere.
    assert sum(len(r["words"]) for r in rows) == len(words)


def test_a_row_with_collapsed_times_is_still_split_by_word_count():
    """Overlapping speech can stamp a whole block at one instant: there is no gap to
    find, so a time-only rule would never split it at all."""
    words = [(0.0, 0.0, f"mot{i}") for i in range(120)]
    raw = [{"t0": 0.0, "t1": 60.0, "speaker": "a", "text": "x", "words": words, "language": "fr"}]
    rows, _ = consolidate(raw, min_speaker_s=0.0, max_s=9.0, max_words=45)
    assert len(rows) >= 3


def test_a_row_without_word_timing_passes_through_untouched():
    """There is nothing to cut on. Guessing a cut point would put a boundary in the
    middle of a word."""
    raw = [
        {
            "t0": 0.0,
            "t1": 60.0,
            "speaker": "a",
            "text": "un long bloc sans mots",
            "words": [],
            "language": "fr",
        }
    ]
    rows, _ = consolidate(raw, min_speaker_s=0.0, max_s=9.0)
    assert len(rows) == 1


# --------------------------------------------------------------------------- #
#  Rule 4 — a row that seeks to silence reads as broken timestamps
# --------------------------------------------------------------------------- #


def test_a_lone_punctuation_row_is_dropped():
    """Sentence splitting on « or . produces a clickable row with no speech in it."""
    raw = [
        _row(0.0, 10.0, "a", "vrai contenu"),
        {"t0": 10.0, "t1": 10.0, "speaker": "b", "text": "»", "words": [], "language": "fr"},
    ]
    rows, _ = consolidate(raw, min_speaker_s=0.0)
    assert [r["text"] for r in rows] == ["vrai contenu"]


def test_a_contentless_row_folds_into_its_same_speaker_neighbour():
    """Folding rather than dropping, when there is a neighbour: the span it covered
    is real even if its text is not."""
    raw = [
        _row(0.0, 5.0, "a", "le début"),
        {"t0": 5.0, "t1": 6.0, "speaker": "a", "text": ".", "words": [], "language": "fr"},
    ]
    rows, _ = consolidate(raw, min_speaker_s=0.0, max_s=100.0)
    assert len(rows) == 1
    assert rows[0]["t1"] == 6.0


def test_a_genuine_short_backchannel_is_kept():
    """ "Ah." and "Ok." are data: who interjects, and when, is exactly what a
    diarized transcript is for. Only rows with fewer than two alphanumerics go."""
    raw = [_row(0.0, 10.0, "a", "une intervention"), _row(10.0, 10.6, "b", "Ok.")]
    rows, _ = consolidate(raw, min_speaker_s=0.0)
    assert "Ok." in [r["text"] for r in rows]


# --------------------------------------------------------------------------- #
#  Rule 5 — a zero-width row can never highlight as it plays
# --------------------------------------------------------------------------- #


def test_a_zero_width_row_is_given_a_span_and_monotonic_word_times():
    raw = [
        {
            "t0": 12.0,
            "t1": 12.0,
            "speaker": "a",
            "text": "C'est ça",
            "words": [(12.0, 12.0, "C'est"), (12.0, 12.0, "ça")],
            "language": "fr",
        },
        _row(30.0, 35.0, "a", "la suite du propos tenu ici"),
    ]
    rows, _ = consolidate(raw, min_speaker_s=0.0, max_s=100.0)
    padded = rows[0]
    assert padded["t1"] > padded["t0"]
    starts = [w[0] for w in padded["words"]]
    assert starts == sorted(starts)
    assert padded["words"][-1][1] <= padded["t1"] + 1e-9


def test_padding_never_runs_into_the_next_row():
    """Two rows highlighting at once reads as a worse bug than the one being fixed."""
    raw = [
        {
            "t0": 10.0,
            "t1": 10.0,
            "speaker": "a",
            "text": "mot " * 20,
            "words": [(10.0, 10.0, "mot")] * 20,
            "language": "fr",
        },
        _row(10.5, 20.0, "b", "la suite immédiate"),
    ]
    rows, _ = consolidate(raw, min_speaker_s=0.0, max_s=100.0)
    assert rows[0]["t1"] < rows[1]["t0"]


# --------------------------------------------------------------------------- #
#  The whole chain
# --------------------------------------------------------------------------- #


def test_an_empty_transcript_consolidates_to_nothing_without_complaining():
    assert consolidate([]) == ([], {})


def test_the_default_floor_is_two_minutes_and_is_documented_as_tunable():
    """The default suits a panel, where a four-word voice is a passer-by. On a
    two-person interview where one person barely speaks it is exactly wrong, which
    is why it is a parameter and why that is said in the docstring."""
    import inspect

    assert inspect.signature(consolidate).parameters["min_speaker_s"].default == 120.0
    assert "Lower it" in (consolidate.__doc__ or "")
