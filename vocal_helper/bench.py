"""
vocal_helper.bench
==================

Transcription accuracy, scored the way a multi-speaker recording actually fails.

Provenance
----------
The French normalisation and the WER report are ported from
``operator-assistance-streaming/streaming/score_wer.py``, which is being absorbed into this
package. They are kept close to the original — the filler list and the repeat-collapsing are
calibrated on real French calls and there is no reason to re-derive them — with one coupling
removed: the original read its reference out of a Gladia JSON, so every number it produced
was *agreement with Gladia* rather than accuracy. A reference here is text, or turns, from
wherever the caller got them.

What plain WER cannot see
-------------------------
On a recording with more than one voice, the failure that destroys everything downstream is
not a misheard word, it is **the right words attributed to the wrong person**. WER is blind to
it: concatenate a whole episode and a diarizer that merged five panellists into two scores
exactly as well as one that separated them. DER is blind the other way — it never reads a
word. :func:`cpwer` prices both at once by scoring under the best possible speaker mapping, so
a transcript only wins if the words AND the speaker are right.

That is not theoretical here. A 2026-10-19 bench on three real sixty-four-minute six-speaker
panels had the local diarization path emitting 103 to 124 clusters for six people. Plain WER on
the concatenated text would barely have moved.

The other thing a WER average hides is which words were missed. Getting "le Figaro" wrong as
"le Fégaro" costs the same fraction of a point as a dropped "the", while destroying any
corpus-level count of who discussed what. :func:`term_error_rate` scores only the terms that
carry that weight — the glossary, the proper nouns — and is the direct measurement of what a
glossary-correction stage exists to fix.

Credits
-------
cpWER is the CHiME-6 formulation (Watanabe et al., 2020). Implemented here rather than
imported so the package keeps its dependency surface; the optimal assignment uses SciPy's
Hungarian solver.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import itertools
import re
import unicodedata

#: Punctuation stripper that keeps apostrophes, so French clitics ("c'est", "qu'il") survive
#: as one token instead of splitting into noise.
PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)
SPACE_RE = re.compile(r"\s+")

#: French disfluencies. Deliberately short: a long list starts eating real content, and the
#: cost of missing one filler is far smaller than the cost of deleting a word somebody said.
FR_FILLERS = {
    "euh", "euhh", "hein", "ben", "bah", "heu", "hum",
    "mh", "mhmh", "mhh", "mmh", "uh", "uhm", "um",
}


def normalize_fr(text: str, *, drop_fillers: bool = False,
                 collapse_repeats: bool = False) -> str:
    """Normalise a French transcript for scoring.

    Always: NFKC, lowercase, punctuation stripped (apostrophes kept), whitespace collapsed.

    ``drop_fillers`` and ``collapse_repeats`` are the style-tolerant mode, and they must be
    applied to **both** sides or the comparison is meaningless. They exist because engines
    disagree about transcription style rather than about what was said: a verbatim model that
    writes every "euh" and every false start is not less accurate than one that tidies them
    away, and scoring them against each other without this punishes honesty.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    if not (drop_fillers or collapse_repeats):
        return text
    tokens = text.split()
    if drop_fillers:
        tokens = [t for t in tokens if t not in FR_FILLERS]
    if collapse_repeats:
        tokens = _collapse_repeats(tokens, max_phrase_len=4)
    return " ".join(tokens)


def _collapse_repeats(tokens: list[str], max_phrase_len: int = 4) -> list[str]:
    """Collapse immediate phrase repetitions of up to *max_phrase_len* tokens.

    Greedy, left to right: at each position take the longest phrase just emitted that the
    next tokens repeat, and skip past the duplicate. Handles the French restart pattern
    ("c'est la c'est la c'est la"), which a speaker produces and a verbatim model faithfully
    writes down.
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        out.append(tokens[i])
        skipped = True
        while skipped:
            skipped = False
            for length in range(min(max_phrase_len, len(out)), 0, -1):
                if tokens[i + 1: i + 1 + length] == out[-length:]:
                    i += length
                    skipped = True
                    break
        i += 1
    return out


def wer_report(reference: str, hypothesis: str, *, style_tolerant: bool = False) -> dict:
    """WER, MER, WIL and CER between two transcripts, plus token counts.

    Returns ``{"wer": None, "error": ...}`` rather than raising on an empty reference: a
    sweep over many recordings must report the hole and carry on, not die on one bad file.
    """
    import jiwer

    ref = normalize_fr(reference, drop_fillers=style_tolerant, collapse_repeats=style_tolerant)
    hyp = normalize_fr(hypothesis, drop_fillers=style_tolerant, collapse_repeats=style_tolerant)
    if not ref:
        return {"wer": None, "error": "empty reference"}
    return {
        "style_tolerant": style_tolerant,
        "wer": round(jiwer.wer(ref, hyp), 4),
        "mer": round(jiwer.mer(ref, hyp), 4),
        "wil": round(jiwer.wil(ref, hyp), 4),
        "cer": round(jiwer.cer(ref, hyp), 4),
        "ref_words": len(ref.split()),
        "hyp_words": len(hyp.split()),
    }


def term_error_rate(reference: str, hypothesis: str, terms: list[str]) -> dict:
    """How many of *terms* that the reference contains the hypothesis also got.

    A WER average treats every word alike, which is wrong in the direction that matters. A
    dropped article costs the same as a mangled proper noun, while only one of the two
    destroys a corpus-level count of who discussed what — "le Fégaro" for *Le Figaro* is a
    rounding error in WER and a lost record in the analysis.

    Multi-word terms are matched as phrases on the normalised text. Only terms actually
    present in the reference count: scoring a hypothesis for missing a word nobody said would
    make the rate depend on the glossary's length rather than on the transcription.

    Returns ``present`` (terms the reference uses), ``found``, ``missed`` (the list, so the
    failure is inspectable and not just a number) and ``error_rate`` in ``[0, 1]``.
    """
    ref = normalize_fr(reference)
    hyp = normalize_fr(hypothesis)
    present, found, missed = 0, 0, []
    for raw in terms:
        term = normalize_fr(str(raw))
        if not term or term not in ref:
            continue
        present += 1
        if term in hyp:
            found += 1
        else:
            missed.append(raw)
    return {
        "present": present,
        "found": found,
        "missed": missed,
        "error_rate": round(1.0 - found / present, 4) if present else None,
    }


def _concat_by_speaker(turns: list[dict]) -> dict[str, str]:
    """``{speaker: all their words, in time order}`` — one stream per voice."""
    buckets: dict[str, list[tuple[float, str]]] = {}
    for t in turns:
        spk = str(t.get("speaker", ""))
        buckets.setdefault(spk, []).append((float(t.get("t0", 0.0)), str(t.get("text") or "")))
    return {
        spk: " ".join(txt for _t0, txt in sorted(items))
        for spk, items in buckets.items()
    }


def cpwer(reference_turns: list[dict], hypothesis_turns: list[dict], *,
          style_tolerant: bool = False) -> dict:
    """Concatenated minimum-permutation WER — words and speaker, priced together.

    Each side is concatenated per speaker into one stream per voice, then speakers are
    matched one-to-one so the total word error is as low as it can be, and the WER of that
    best case is reported. A transcript therefore only scores well when the words are right
    **and** they are under the right person.

    This is the number to quote for a multi-speaker recording. Plain WER on concatenated text
    cannot tell a diarizer that separated five panellists from one that merged them into two;
    DER can, but never reads a word; cpWER refuses both blind spots.

    Speaker counts need not match. Unmatched speakers — on either side — are charged in full,
    which is the honest treatment: inventing a sixth voice and missing a real one are both
    errors, and a metric that quietly dropped the leftovers would reward over-splitting.

    Returns the ``cpwer`` figure, the chosen ``mapping`` (hypothesis speaker → reference
    speaker), and both speaker counts so a bad score can be read as mis-transcription or as
    mis-attribution without re-running anything. ``n_ref_speakers`` vs ``n_hyp_speakers`` is
    often the whole story.
    """
    import jiwer

    ref_by = _concat_by_speaker(reference_turns)
    hyp_by = _concat_by_speaker(hypothesis_turns)
    norm = lambda s: normalize_fr(  # noqa: E731 - one binding, used twice below
        s, drop_fillers=style_tolerant, collapse_repeats=style_tolerant
    )
    ref_by = {k: norm(v) for k, v in ref_by.items() if norm(v)}
    hyp_by = {k: norm(v) for k, v in hyp_by.items()}
    if not ref_by:
        return {"cpwer": None, "error": "empty reference"}

    ref_keys = sorted(ref_by)
    hyp_keys = sorted(hyp_by)
    ref_words = sum(len(v.split()) for v in ref_by.values())

    def _errors(ref_text: str, hyp_text: str) -> float:
        """Word errors, not a rate: rates are not additive across speakers, counts are."""
        out = jiwer.process_words(ref_text, hyp_text)
        return float(out.substitutions + out.deletions + out.insertions)

    # Cost of pairing each reference speaker with each hypothesis speaker, plus the cost of
    # leaving either unmatched (deleting / inserting their whole stream).
    unmatched_ref = {k: float(len(v.split())) for k, v in ref_by.items()}
    unmatched_hyp = {k: float(len(v.split())) for k, v in hyp_by.items()}

    best_cost, best_map = None, {}
    if ref_keys and hyp_keys:
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment

            size = max(len(ref_keys), len(hyp_keys))
            cost = np.zeros((size, size), dtype=float)
            for i in range(size):
                for j in range(size):
                    r = ref_keys[i] if i < len(ref_keys) else None
                    h = hyp_keys[j] if j < len(hyp_keys) else None
                    if r is None and h is None:
                        cost[i, j] = 0.0
                    elif r is None:
                        cost[i, j] = unmatched_hyp[h]
                    elif h is None:
                        cost[i, j] = unmatched_ref[r]
                    else:
                        cost[i, j] = _errors(ref_by[r], hyp_by[h])
            rows, cols = linear_sum_assignment(cost)
            best_cost = float(cost[rows, cols].sum())
            best_map = {
                hyp_keys[j]: ref_keys[i]
                for i, j in zip(rows, cols)
                if i < len(ref_keys) and j < len(hyp_keys)
            }
        except ImportError:
            # Exhaustive fallback. Factorial, so it is capped: past a handful of speakers the
            # Hungarian solver is the only sane route, and silently returning a wrong answer
            # would be worse than saying so.
            if max(len(ref_keys), len(hyp_keys)) > 7:
                return {"cpwer": None, "error": "scipy required for >7 speakers"}
            for perm in itertools.permutations(hyp_keys, min(len(ref_keys), len(hyp_keys))):
                total = 0.0
                used = set()
                for r, h in zip(ref_keys, perm):
                    total += _errors(ref_by[r], hyp_by[h])
                    used.add(h)
                total += sum(unmatched_ref[r] for r in ref_keys[len(perm):])
                total += sum(v for k, v in unmatched_hyp.items() if k not in used)
                if best_cost is None or total < best_cost:
                    best_cost = total
                    best_map = dict(zip(perm, ref_keys))
    else:
        best_cost = sum(unmatched_ref.values()) + sum(unmatched_hyp.values())

    return {
        "cpwer": round(best_cost / ref_words, 4) if ref_words else None,
        "errors": int(best_cost) if best_cost is not None else None,
        "ref_words": ref_words,
        "n_ref_speakers": len(ref_keys),
        "n_hyp_speakers": len(hyp_keys),
        "mapping": best_map,
        "style_tolerant": style_tolerant,
    }


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Seconds two intervals share."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def speaker_label_error(reference_turns: list[dict], hypothesis_turns: list[dict]) -> dict:
    """How often the speaker label on a displayed turn is wrong.

    Ported from ``operator-assistance-streaming/streaming/score_der.py`` and generalised off
    its Gladia-only reference. For each reference turn, the hypothesis speaker holding the
    most overlap with it wins the vote; labels are then matched one-to-one so the global
    error is as low as possible, and the remaining mismatches are counted.

    The point is what it *ignores*: boundary slop. DER charges every millisecond at the edge
    of a turn, which is right for comparing diarizers and wrong for a product — a reader
    never sees a boundary, they see a name against a paragraph. This asks their question
    instead: how often is that name wrong.

    Returns ``turns``, ``errors``, ``error_rate``, the chosen ``mapping`` and
    ``unattributed`` (reference turns no hypothesis speaker overlaps at all, counted as
    errors because a missing label is a wrong label to the person reading it).
    """
    ref = [t for t in reference_turns if float(t.get("t1", 0)) > float(t.get("t0", 0))]
    if not ref:
        return {"turns": 0, "errors": 0, "error_rate": None, "mapping": {}, "unattributed": 0}

    votes: list[tuple[str, str | None]] = []
    unattributed = 0
    for r in ref:
        r0, r1 = float(r["t0"]), float(r["t1"])
        best, best_ov = None, 0.0
        for h in hypothesis_turns:
            ov = _overlap(r0, r1, float(h.get("t0", 0)), float(h.get("t1", 0)))
            if ov > best_ov:
                best, best_ov = str(h.get("speaker", "")), ov
        if best is None:
            unattributed += 1
        votes.append((str(r.get("speaker", "")), best))

    ref_labels = sorted({r for r, _h in votes})
    hyp_labels = sorted({h for _r, h in votes if h is not None})
    counts: dict[tuple[str, str], int] = {}
    for r, h in votes:
        if h is not None:
            counts[(r, h)] = counts.get((r, h), 0) + 1

    mapping: dict[str, str] = {}
    if ref_labels and hyp_labels:
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment

            size = max(len(ref_labels), len(hyp_labels))
            # Maximise agreement ⇒ minimise its negative.
            gain = np.zeros((size, size), dtype=float)
            for i, r in enumerate(ref_labels):
                for j, h in enumerate(hyp_labels):
                    gain[i, j] = -counts.get((r, h), 0)
            rows, cols = linear_sum_assignment(gain)
            mapping = {
                hyp_labels[j]: ref_labels[i]
                for i, j in zip(rows, cols)
                if i < len(ref_labels) and j < len(hyp_labels)
            }
        except ImportError:
            # Greedy fallback: take the best remaining pair until one side runs out. Not
            # optimal, and said so rather than pretending otherwise.
            taken_r, taken_h = set(), set()
            for (r, h), _n in sorted(counts.items(), key=lambda kv: -kv[1]):
                if r not in taken_r and h not in taken_h:
                    mapping[h] = r
                    taken_r.add(r)
                    taken_h.add(h)

    errors = sum(1 for r, h in votes if h is None or mapping.get(h) != r)
    return {
        "turns": len(votes),
        "errors": errors,
        "error_rate": round(errors / len(votes), 4),
        "mapping": mapping,
        "unattributed": unattributed,
    }


def der_report(reference_turns: list[dict], hypothesis_turns: list[dict], *,
               collar: float = 0.0, skip_overlap: bool = False) -> dict:
    """DER broken into its parts, plus JER and both speaker counts.

    The defaults are **collar 0 and overlap scored**, which is not the convention. The usual
    0.25 s collar with overlap skipped deletes exactly the regions an interruption study is
    about, and a system can look good on it precisely by ignoring simultaneous speech. Pass
    ``collar=0.25, skip_overlap=True`` for a number comparable to published work, and read it
    as a second line rather than the verdict.

    Why the decomposition is not optional: miss, false alarm and confusion have different
    cures. A high false alarm is a voice-activity problem, a high confusion is an identity
    problem, and a single DER cannot tell you which one you have. On the 2026-10-19 AMI
    bench the two local paths had near-identical DERs and opposite failures.

    ``jer`` is reported next to it because DER is time-weighted: one dominant speaker can
    hide the collapse of everyone else, which is the failure mode of a panel show. JER
    normalises per speaker and does not let that through.

    ``n_ref_speakers`` / ``n_hyp_speakers`` are there because a count is the bluntest and
    often the most informative signal — a model with an architectural ceiling cannot emit a
    fifth label, and that shows here before any error rate is read.

    Requires the ``pyannote`` extra for the metrics themselves.
    """
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    def _ann(turns: list[dict]) -> Annotation:
        ann = Annotation()
        for i, t in enumerate(turns):
            t0, t1 = float(t.get("t0", 0.0)), float(t.get("t1", 0.0))
            if t1 > t0:
                ann[Segment(t0, t1), i] = str(t.get("speaker", ""))
        return ann

    ref, hyp = _ann(reference_turns), _ann(hypothesis_turns)
    if not ref.labels():
        return {"der": None, "error": "empty reference"}

    der = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    jer = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
    detail = der(ref, hyp, detailed=True)
    total = max(1e-9, detail["total"])
    return {
        "der": round(float(detail["diarization error rate"]), 4),
        "miss": round(float(detail["missed detection"]) / total, 4),
        "false_alarm": round(float(detail["false alarm"]) / total, 4),
        "confusion": round(float(detail["confusion"]) / total, 4),
        "jer": round(float(jer(ref, hyp)), 4),
        "n_ref_speakers": len(ref.labels()),
        "n_hyp_speakers": len(hyp.labels()),
        "collar": collar,
        "skip_overlap": skip_overlap,
    }
