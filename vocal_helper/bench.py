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
