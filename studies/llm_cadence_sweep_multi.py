"""LLM rolling-summary cadence sweep — multi-meeting variant.

Same protocol as ``llm_cadence_sweep.py`` (RTF + cosine-sim vs an
offline single-shot reference) but runs across **multiple AMI
dev-slice meetings** so the recommended operating point is robust to
the noise of any one conversation.

Why this and not ``llm_cadence_sweep.py``
-----------------------------------------
The single-meeting sweep gives a quick directional answer but is at
the mercy of one conversation's structure (topic density, speaker
turn cadence, length). The pdbms canonical studies pin everything at
N ≥ 4 meetings before quoting a result ; this script does the same for
the LLM-analyst cadence.

Corpus — AMI dev-slice
----------------------
``IS1008a``  — 16 min,  4 speakers, technical project kickoff (anchor).
``ES2011a``  — 19 min,  4 speakers, design meeting #1.
``ES2011d``  — 33 min,  4 speakers, design meeting #4 (dense long).
``TS3004a``  — 22 min,  4 speakers, telephone-style meeting.

Four meetings span short / medium / long durations and the two main
recording styles. Median over the four gives the recommended setting.

Author : Warith HARCHAOUI — 2026-06-30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics

# Re-use the single-meeting helpers — no duplication.
import sys
from pathlib import Path

# Put this study's own directory on ``sys.path`` so the sibling
# single-meeting module can be imported as a top-level name below.
_STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STUDY_DIR))
# E402: this import must follow the ``sys.path`` tweak above, so it can't sit
# with the other top-level imports — reuse rather than duplicate the helpers.
from llm_cadence_sweep import (  # type: ignore  # noqa: E402
    candidate_summary,
    cosine_sim,
    load_utterances,
    reference_summary,
)

# Corpus root and the four dev-slice meetings spanning short/medium/long and
# both recording styles; the median over these is what we quote.
DEFAULT_AMI = Path("/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice")
MEETINGS = ["IS1008a", "ES2011a", "ES2011d", "TS3004a"]
DEFAULT_MODEL = "gemma4:e4b-mlx"
# Run log; the JSON dump is derived from it via ``.with_suffix(".json")``.
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/vocal_helper_llm_cadence_multi_2026-06-30.log"
)

# Multi-meeting variant — focused on the three configs that emerged
# from the single-meeting sweep as the Pareto front (t=60s = max
# cos_sim 0.420 ; t=120s = min RTF 0.192 ; n=20 = best n-based, RTF
# 0.260 / cos_sim 0.397). Running these three against 4 meetings is
# tractable (~ 30-40 min total) and answers : is t=60s the winner on
# IS1008a alone, or does it hold up across meeting styles ?
CONFIGS: list[tuple[str, int, float | None]] = [
    ("n=20", 20, None),
    ("t=60s", 10_000, 60.0),
    ("t=120s", 10_000, 120.0),
]


async def amain(args: argparse.Namespace) -> None:
    """Run the cadence sweep across all meetings and pool the results.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments carrying ``model`` (Ollama model tag) and ``host``
        (optional Ollama host URL). The corpus is fixed via ``MEETINGS``.

    Returns
    -------
    None
        Per-meeting rows, the pooled (median-over-meetings) table and the
        winner are emitted to stdout / the log / a sibling JSON dump.
    """
    # Start from a clean log so each multi-meeting run stands on its own.
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")

    def log(msg: str) -> None:
        """Echo a line to stdout and append it to the run log.

        Parameters
        ----------
        msg : str
            The line to emit (without trailing newline).

        Returns
        -------
        None
            Writes to stdout and to ``DEFAULT_LOG`` as a side effect.
        """
        # ``print`` is the documented study-result channel; the file copy is a
        # durable record of the run.
        print(msg, flush=True)
        with open(DEFAULT_LOG, "a") as f:
            f.write(msg + "\n")

    log("# LLM cadence sweep — multi-meeting — 2026-06-30")
    log(f"# model    : {args.model}")
    log(f"# meetings : {MEETINGS}")

    # Per-meeting → per-config → (wall, rtf, n_calls, cos_sim), plus the audio
    # span of each meeting for the RTF denominators and the JSON dump.
    per_meeting_results: dict[str, dict[str, tuple[float, float, int, float]]] = {}
    durations: dict[str, float] = {}

    for m in MEETINGS:
        # Skip meetings whose RTTM is absent so a missing slice doesn't abort
        # the whole multi-meeting run.
        rttm = DEFAULT_AMI / m / "words.rttm"
        if not rttm.exists():
            log(f"\n{m}: missing rttm, skipping")
            continue

        # Load the transcript and record its audio span for RTF.
        utts = load_utterances(rttm)
        dur = max(u["t1"] for u in utts) - min(u["t0"] for u in utts)
        durations[m] = dur
        log(f"\n{m}  n_utt={len(utts)}  dur={dur:.0f}s")

        # Reference once per meeting — the oracle each candidate is scored on.
        ref_text, ref_wall = reference_summary(utts, args.model, args.host)
        log(f"  reference  wall={ref_wall:6.1f}s  RTF={ref_wall / dur:.3f}  chars={len(ref_text)}")

        # Score each Pareto-front cadence for this meeting.
        per_meeting_results[m] = {}
        for label, fn, fs in CONFIGS:
            # Fixed 60 s recent window; only the flush cadence varies.
            text, wall, n_calls = await candidate_summary(
                utts,
                model=args.model,
                host=args.host,
                recent_window_s=60.0,
                flush_every_n=fn,
                flush_every_s=fs,
            )
            # Quality vs oracle and RTF vs this meeting's span.
            sim = cosine_sim(text, ref_text)
            rtf = wall / dur
            per_meeting_results[m][label] = (wall, rtf, n_calls, sim)
            log(
                f"  {label:<10s}  wall={wall:>7.1f}  RTF={rtf:>6.3f}  "
                f"n={n_calls:>4d}  cos={sim:>5.3f}"
            )

    # ----- pooled summary -----
    # Print the pooled table header before folding per-meeting rows into medians.
    log("\n" + "=" * 80)
    log("Pooled summary (median over meetings)")
    log("=" * 80)
    log(f"{'config':<10s}  {'med_RTF':>8s}  {'med_n':>6s}  {'med_cos':>8s}")
    log("-" * 40)
    # For each cadence, collapse its per-meeting metrics to a median so one
    # noisy meeting can't dominate the recommendation.
    pooled: dict[str, tuple[float, int, float]] = {}
    for label, _, _ in CONFIGS:
        # Gather this config's metric across the meetings that actually ran
        # (the ``if v`` filters out meetings skipped for a missing RTTM).
        rtfs = [v[1] for m in MEETINGS for v in [per_meeting_results.get(m, {}).get(label)] if v]
        ns = [v[2] for m in MEETINGS for v in [per_meeting_results.get(m, {}).get(label)] if v]
        sims = [v[3] for m in MEETINGS for v in [per_meeting_results.get(m, {}).get(label)] if v]
        # No data for this config (all meetings skipped) → nothing to pool.
        if not rtfs:
            continue
        # Median across meetings is the robust central tendency we report.
        m_rtf = statistics.median(rtfs)
        m_n = int(statistics.median(ns))
        m_sim = statistics.median(sims)
        pooled[label] = (m_rtf, m_n, m_sim)
        log(f"{label:<10s}  {m_rtf:>8.3f}  {m_n:>6d}  {m_sim:>8.3f}")

    # Pick : max median cos_sim with median RTF ≤ 0.10.
    # Keep only cadences within the real-time RTF budget first.
    feasible = [(k, v) for k, v in pooled.items() if v[0] <= 0.10]
    # If none fit the budget, rank all pooled configs so we still name a winner.
    if not feasible:
        feasible = list(pooled.items())
    winner = max(feasible, key=lambda kv: kv[1][2])
    log(
        f"\nWinner : {winner[0]}  "
        f"(median RTF {winner[1][0]:.3f}, n {winner[1][1]}, "
        f"cos_sim {winner[1][2]:.3f})"
    )

    # Persist the full result matrix (pooled + per-meeting) for downstream use.
    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "model": args.model,
                "meetings": MEETINGS,
                "durations_s": durations,
                "pooled": pooled,
                "per_meeting": {
                    m: {label: list(v) for label, v in per_meeting_results.get(m, {}).items()}
                    for m in MEETINGS
                },
                "winner": winner[0],
            },
            indent=2,
        )
    )
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


def main() -> None:
    """Parse CLI arguments and run the multi-meeting cadence sweep.

    Returns
    -------
    None
        Delegates to :func:`amain` under the asyncio event loop; all output is
        side-effect (stdout / log / JSON).
    """
    # Model and host only; the meeting corpus is fixed by ``MEETINGS``.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=None)
    args = p.parse_args()
    # The sweep is async (queue-driven analyst); run it to completion.
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
