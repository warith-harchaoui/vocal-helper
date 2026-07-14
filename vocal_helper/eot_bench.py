"""
vocal_helper.eot_bench
======================

End-of-turn detection benchmark utilities — reusable evaluation
functions inspired by LiveKit's ``eot-bench`` methodology
[@livekitturndetector] (April 2026 whitepaper).

Motivation
----------
Every EOT / turn-taking study we run needs the same primitives :

- A ground-truth turn boundary (the moment speaker A actually
  stops holding the floor).
- A detector's commit boundary (the moment the detector emits
  "turn ended, hand over").
- A latency band under which we score the detector (e.g. 300 ms,
  600 ms, 1200 ms).
- A pair of failure modes to count :
  * **false-cutoff** — the detector fired *before* the true turn
    end (interrupts the speaker prematurely).
  * **hang** — the detector waited too long *after* the true turn
    end (perceived as agent hesitation).

This module implements both metrics as pure functions of aligned
lists of ``(true_turn_end_s, detector_commit_s)`` pairs. Every
call is deterministic ; no models are loaded here. The consumer
brings the ground truth (from RTTM / manual annotation) and the
detector output ; this module scores them.

Origin — the LiveKit paper reports :

- 9.9 % false-cutoff at 300 ms median semantic latency
- 4.5 % false-cutoff at 600 ms
- 3.0 % false-cutoff at 1200 ms

on their proprietary conversational corpus. We adopt the same
three latency bands as the default reporting protocol.

Public surface
--------------

- :class:`EOTPair` — one (true_turn_end, detector_commit) datum.
- :func:`false_cutoff_rate` — fraction of pairs where
  ``commit < true_end - tolerance``.
- :func:`hang_rate` — fraction of pairs where
  ``commit > true_end + tolerance``.
- :func:`score` — full breakdown at the standard {300, 600, 1200}
  ms latency bands.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# LiveKit's canonical reporting bands — 300 / 600 / 1200 ms — kept as the
# module default so every study scores against the same latency budgets.
DEFAULT_LATENCY_BANDS_MS: tuple[int, ...] = (300, 600, 1200)


@dataclass(frozen=True)
class EOTPair:
    """One aligned ground-truth / detector datum.

    Attributes
    ----------
    true_turn_end_s
        Absolute (mix-time) second at which the speaker actually
        released the floor. In practice pulled from a hand-labelled
        turn boundary or a bridged word-level RTTM.
    detector_commit_s
        Absolute second at which the detector emitted its
        end-of-turn decision. For a semantic-EOT model this is
        typically ``true_turn_end + inference_latency`` on hits.
    """

    true_turn_end_s: float
    detector_commit_s: float


def _latency_s(pair: EOTPair) -> float:
    """Signed latency : positive = detector fired *after* true end."""
    # Sign convention is the whole point : negative ⇒ premature cutoff,
    # positive ⇒ hang. Both metrics below key off this single subtraction.
    return pair.detector_commit_s - pair.true_turn_end_s


def false_cutoff_rate(
    pairs: Sequence[EOTPair],
    *,
    tolerance_s: float,
) -> float:
    """Share of pairs where the detector fired ``tolerance_s`` early.

    A "false cutoff" is a decision to end the turn while the
    speaker is still speaking — the perceptual failure LiveKit's
    metric targets. Setting ``tolerance_s = 0`` treats *any*
    early commit as a false cutoff ; realistic tolerances are
    50-150 ms (below human turn-taking noise floor).
    """
    if not pairs:
        return 0.0
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")
    n_early = sum(1 for p in pairs if _latency_s(p) < -tolerance_s)
    return n_early / len(pairs)


def hang_rate(
    pairs: Sequence[EOTPair],
    *,
    latency_budget_s: float,
) -> float:
    """Share of pairs where the detector waited > ``latency_budget_s``.

    The other-side failure : the speaker finished but the agent
    kept waiting, producing perceived agent hesitation.
    ``latency_budget_s`` is the acceptable upper bound (LiveKit
    uses 300 / 600 / 1200 ms bands).
    """
    if not pairs:
        return 0.0
    if latency_budget_s < 0:
        raise ValueError("latency_budget_s must be non-negative")
    n_hang = sum(1 for p in pairs if _latency_s(p) > latency_budget_s)
    return n_hang / len(pairs)


def median_latency_s(pairs: Sequence[EOTPair]) -> float:
    """Median signed latency across the corpus."""
    import statistics

    if not pairs:
        return 0.0
    return statistics.median(_latency_s(p) for p in pairs)


def score(
    pairs: Iterable[EOTPair],
    *,
    latency_bands_ms: Sequence[int] = DEFAULT_LATENCY_BANDS_MS,
    tolerance_ms: int = 50,
) -> dict:
    """Full LiveKit-style report on a set of paired detections.

    Parameters
    ----------
    pairs
        Iterable of aligned (true_end, detector_commit) pairs.
    latency_bands_ms
        Latency budgets to score the hang rate against. Default is
        LiveKit's canonical (300, 600, 1200) ms bands.
    tolerance_ms
        Acceptable early-commit slack before we count a false-cutoff.
        Default 50 ms sits below human turn-taking noise floor
        (Heldner & Edlund 2010 median gap 200 ms).

    Returns
    -------
    dict
        ``{"n": int, "median_latency_ms": float,
        "false_cutoff_rate": float,
        "hang_rate_at_ms": {band: rate, ...}}``.
    """
    pairs = list(pairs)
    return {
        "n": len(pairs),
        "median_latency_ms": median_latency_s(pairs) * 1000.0,
        "false_cutoff_rate": false_cutoff_rate(
            pairs,
            tolerance_s=tolerance_ms / 1000.0,
        ),
        "hang_rate_at_ms": {
            band: hang_rate(pairs, latency_budget_s=band / 1000.0) for band in latency_bands_ms
        },
    }
