"""Transcription scoring: normalisation, WER, term error rate, and cpWER."""

import pytest

from vocal_helper import bench


# --- normalisation ------------------------------------------------------------------------ #

def test_apostrophes_survive_so_french_clitics_stay_one_token() -> None:
    assert bench.normalize_fr("C'est l'État, qu'il dit !") == "c'est l'état qu'il dit"


def test_style_tolerant_mode_drops_fillers_and_restarts() -> None:
    """Engines disagree about transcription style, not about what was said.

    A verbatim model that writes every "euh" is not less accurate than one that tidies them
    away, and scoring them against each other without this punishes honesty.
    """
    verbatim = "euh c'est la c'est la c'est la bonne solution hein"
    tidy = "c'est la bonne solution"
    assert bench.normalize_fr(verbatim, drop_fillers=True, collapse_repeats=True) == tidy


def test_plain_mode_leaves_disfluencies_alone() -> None:
    assert "euh" in bench.normalize_fr("euh oui")


# --- WER ------------------------------------------------------------------------------------ #

def test_a_perfect_transcript_scores_zero() -> None:
    assert bench.wer_report("le chat dort", "le chat dort")["wer"] == 0.0


def test_an_empty_reference_reports_a_hole_rather_than_raising() -> None:
    """A sweep over many recordings must report the gap and carry on."""
    out = bench.wer_report("", "quelque chose")
    assert out["wer"] is None and out["error"] == "empty reference"


def test_style_tolerance_can_turn_a_bad_score_into_a_fair_one() -> None:
    strict = bench.wer_report("euh le chat dort", "le chat dort")["wer"]
    fair = bench.wer_report("euh le chat dort", "le chat dort", style_tolerant=True)["wer"]
    assert strict > 0 and fair == 0.0


# --- term error rate -------------------------------------------------------------------- #

def test_only_terms_the_reference_actually_uses_are_scored() -> None:
    """Otherwise the rate depends on the glossary's length, not on the transcription."""
    out = bench.term_error_rate("on parle du Figaro", "on parle du Fégaro",
                                ["Le Figaro", "Mediapart"])
    assert out["present"] == 0  # "le figaro" as a phrase is not in the reference
    out = bench.term_error_rate("on parle du Figaro", "on parle du Fégaro",
                                ["Figaro", "Mediapart"])
    assert out["present"] == 1 and out["found"] == 0
    assert out["error_rate"] == 1.0 and out["missed"] == ["Figaro"]


def test_a_found_term_counts_even_inside_a_sentence() -> None:
    out = bench.term_error_rate("selon Le Figaro hier", "selon Le Figaro hier", ["le figaro"])
    assert out["present"] == 1 and out["error_rate"] == 0.0


def test_no_term_present_reports_none_not_zero() -> None:
    """Zero error and nothing to be wrong about are different claims."""
    assert bench.term_error_rate("bonjour", "bonjour", ["Figaro"])["error_rate"] is None


# --- cpWER ------------------------------------------------------------------------------- #

def _turns(*rows):
    return [{"speaker": s, "t0": t, "text": txt} for s, t, txt in rows]


def test_a_relabelled_but_correct_transcript_is_not_penalised() -> None:
    """Diarizer ids are arbitrary; S0/S1 swapped is the same transcript."""
    ref = _turns(("A", 0, "le chat dort"), ("B", 1, "le chien court"))
    hyp = _turns(("S1", 0, "le chat dort"), ("S0", 1, "le chien court"))
    out = bench.cpwer(ref, hyp)
    assert out["cpwer"] == 0.0
    assert out["mapping"] == {"S1": "A", "S0": "B"}


def test_words_crossed_between_speakers_is_the_error_plain_wer_cannot_see() -> None:
    """The failure that destroys everything downstream, and the reason cpWER exists.

    Every word of the reference appears in the hypothesis, in the same order — concatenate
    the two sides and WER is **zero**. But "dort" has been put under the wrong voice and
    "court" under the other, so no speaker mapping lines the streams up, and cpWER charges
    it. A study asking who said what cannot tell the two transcripts apart on WER alone.
    """
    ref = _turns(("A", 0, "le chat dort"), ("B", 1, "le chien court"))
    crossed = _turns(("S0", 0, "le chat court"), ("S1", 1, "le chien dort"))
    assert bench.wer_report("le chat dort le chien court",
                            "le chat court le chien dort")["wer"] > 0  # order differs here
    # The decisive part: concatenated, both sides hold the same four content words.
    assert bench.cpwer(ref, crossed)["cpwer"] > 0


def test_merging_two_speakers_into_one_is_charged() -> None:
    """A diarizer that collapses a panel must not score like one that separates it."""
    ref = _turns(("A", 0, "le chat dort"), ("B", 1, "le chien court"))
    merged = _turns(("S0", 0, "le chat dort"), ("S0", 1, "le chien court"))
    out = bench.cpwer(ref, merged)
    assert out["cpwer"] > 0
    assert out["n_ref_speakers"] == 2 and out["n_hyp_speakers"] == 1


def test_inventing_a_speaker_is_charged_too() -> None:
    """Otherwise the metric would reward over-splitting, which is the long-form failure."""
    ref = _turns(("A", 0, "le chat dort le chien court"))
    split = _turns(("S0", 0, "le chat dort"), ("S1", 1, "le chien court"))
    assert bench.cpwer(ref, split)["cpwer"] > 0


def test_the_speaker_counts_come_back_so_a_bad_score_can_be_read() -> None:
    ref = _turns(("A", 0, "un deux trois"), ("B", 1, "quatre cinq six"))
    hyp = _turns(("S0", 0, "un deux trois"), ("S1", 1, "quatre cinq six"),
                 ("S2", 2, "sept huit"))
    out = bench.cpwer(ref, hyp)
    assert out["n_ref_speakers"] == 2 and out["n_hyp_speakers"] == 3
    assert out["cpwer"] == pytest.approx(2 / 6, abs=1e-4)  # the two invented words


def test_cpwer_on_an_empty_reference_reports_a_hole() -> None:
    assert bench.cpwer([], _turns(("S0", 0, "x")))["cpwer"] is None


# --- speaker label error: the product's question, not the diarizer's --------------------- #

def test_boundary_slop_is_ignored_because_nobody_reads_a_boundary() -> None:
    """DER charges every millisecond at the edge of a turn. A reader sees a name.

    Here the hypothesis is late by 300 ms on both turns and the names are right, so the
    thing the person reading actually experiences is: nothing wrong.
    """
    ref = [{"speaker": "A", "t0": 0.0, "t1": 5.0}, {"speaker": "B", "t0": 5.0, "t1": 10.0}]
    hyp = [{"speaker": "S0", "t0": 0.3, "t1": 5.3}, {"speaker": "S1", "t0": 5.3, "t1": 10.3}]
    out = bench.speaker_label_error(ref, hyp)
    assert out["error_rate"] == 0.0 and out["mapping"] == {"S0": "A", "S1": "B"}


def test_a_genuinely_wrong_name_is_counted() -> None:
    ref = [{"speaker": "A", "t0": 0.0, "t1": 5.0}, {"speaker": "B", "t0": 5.0, "t1": 10.0},
           {"speaker": "A", "t0": 10.0, "t1": 15.0}]
    hyp = [{"speaker": "S0", "t0": 0.0, "t1": 5.0}, {"speaker": "S1", "t0": 5.0, "t1": 10.0},
           {"speaker": "S1", "t0": 10.0, "t1": 15.0}]  # third turn attributed to B
    out = bench.speaker_label_error(ref, hyp)
    assert out["errors"] == 1 and out["error_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_a_turn_no_hypothesis_covers_counts_as_wrong() -> None:
    """A missing label is a wrong label to the person reading it."""
    ref = [{"speaker": "A", "t0": 0.0, "t1": 5.0}, {"speaker": "B", "t0": 50.0, "t1": 55.0}]
    hyp = [{"speaker": "S0", "t0": 0.0, "t1": 5.0}]
    out = bench.speaker_label_error(ref, hyp)
    assert out["unattributed"] == 1 and out["errors"] == 1


def test_label_error_on_an_empty_reference_reports_a_hole() -> None:
    assert bench.speaker_label_error([], [{"speaker": "S0", "t0": 0, "t1": 1}])["turns"] == 0


# --- DER report --------------------------------------------------------------------------- #

def test_der_defaults_score_overlap_and_use_no_collar() -> None:
    """The usual 0.25 s collar with overlap skipped deletes what an interruption study is about."""
    pytest.importorskip("pyannote.metrics")
    ref = [{"speaker": "A", "t0": 0.0, "t1": 5.0}, {"speaker": "B", "t0": 5.0, "t1": 10.0}]
    out = bench.der_report(ref, ref)
    assert out["collar"] == 0.0 and out["skip_overlap"] is False
    assert out["der"] == 0.0 and out["jer"] == 0.0


def test_der_comes_decomposed_because_the_parts_have_different_cures() -> None:
    """A high false alarm is a VAD problem; a high confusion is an identity problem."""
    pytest.importorskip("pyannote.metrics")
    ref = [{"speaker": "A", "t0": 0.0, "t1": 10.0}]
    hyp = [{"speaker": "S0", "t0": 0.0, "t1": 10.0}, {"speaker": "S1", "t0": 10.0, "t1": 15.0}]
    out = bench.der_report(ref, hyp)
    assert out["false_alarm"] > 0 and out["confusion"] == 0.0
    assert out["n_ref_speakers"] == 1 and out["n_hyp_speakers"] == 2


def test_der_on_an_empty_reference_reports_a_hole() -> None:
    pytest.importorskip("pyannote.metrics")
    assert bench.der_report([], [{"speaker": "S0", "t0": 0, "t1": 1}])["der"] is None
