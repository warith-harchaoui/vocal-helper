"""LLM model-size & family sweep — RTF and quality across local LLMs.

Goal
----
Picking ``gemma4:e4b`` was the user-spec default ; this study answers
whether a different local LLM produces a better RTF / quality
trade-off for the rolling-summary task.

Tested models (all served by Ollama, all local) :

- ``gemma4:e2b-mlx``  — Gemma 4, 2B effective, MLX (Apple-Silicon).
- ``gemma4:e4b-mlx``  — Gemma 4, 4B effective, MLX. **Current default.**
- ``gemma4:12b-mlx``  — Gemma 4, 12B, MLX.
- ``qwen2.5:3b``      — Qwen 2.5, 3B, gguf.
- ``qwen3:8b``        — Qwen 3, 8B, gguf.

Same protocol as ``llm_cadence_sweep.py`` (single-meeting variant) :

- corpus       : AMI IS1008a (256 utterances, 869 s, 4 speakers)
- cadence      : ``flush_every_s = 60`` (the 2026-06-30 cadence winner)
- recent window: 60 s
- reference    : single LLM call on the full transcript, **per model**
  (so each candidate is judged against ITS OWN best-shot summary).
- metric       : TF-IDF cosine similarity vs the reference + RTF.

Why per-model reference
-----------------------
Different models have different writing styles. Comparing all
candidates against a *single* reference would favour the model that
produced the reference. Using the same model for the reference and
its own candidate isolates the cadence-vs-quality variable from the
inter-model style noise.

Author : Warith HARCHAOUI — 2026-06-30
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import click

# ``GemmaAnalystStage`` / ``_extract_response_text`` used to be imported here
# and were never referenced — the helpers below come from the sibling
# single-meeting module. The second name no longer exists in
# ``vocal_helper.llm`` (the best-engine engine contract removed it), so a
# dead import at module level would break even ``--help``.
# Re-use single-meeting helpers.
_STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STUDY_DIR))
from llm_cadence_sweep import (  # type: ignore
    cosine_sim,
    load_utterances,
    reference_summary,
    candidate_summary,
)

DEFAULT_RTTM = Path(
    "/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice/IS1008a/words.rttm"
)
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/vocal_helper_llm_model_size_2026-06-30.log"
)

MODELS = [
    "gemma4:e2b-mlx",  # MLX, Apple-Silicon native
    "gemma4:e4b-mlx",  # MLX — current default
    "gemma4:12b-mlx",  # MLX, larger
    "gemma3:4b",  # gguf, prior Gemma generation reference
    "qwen2.5:3b",  # gguf, Qwen 2.5 generation
    "qwen3:8b",  # gguf, Qwen 3 generation
    "llama3.2:3b",  # gguf, Llama 3.2 generation
]


def log(msg: str) -> None:
    """Echo a line to stdout and append it to the on-disk study log.

    Parameters
    ----------
    msg : str
        The line to emit. It is printed verbatim (this is a ``studies/``
        result-table line, an allowed exception to the no-print rule) and
        also appended — with a trailing newline — to :data:`DEFAULT_LOG` so
        the run leaves a durable transcript.

    Returns
    -------
    None
        The message is emitted for its side effects only.
    """
    # Console first so a human watching the run sees progress live.
    print(msg, flush=True)
    # Then persist the exact same line so the log file mirrors stdout.
    with open(DEFAULT_LOG, "a") as f:
        f.write(msg + "\n")


async def amain(*, rttm: Path, host: str | None) -> None:
    """Run the full model sweep and write the result table + JSON dump.

    For every candidate in :data:`MODELS` this builds a per-model reference
    summary, streams a rolling candidate summary at the canonical cadence,
    scores the candidate against its own reference, and records RTF. The
    per-model results are then tabulated, reduced to a Pareto frontier, and
    serialised to a sibling ``.json`` file.

    Parameters
    ----------
    rttm : Path
        RTTM transcript to load utterances from.
    host : str or None
        Ollama host URL, or ``None`` for the default endpoint.

    Returns
    -------
    None
        All output is side-effecting: the study log, stdout, and the JSON
        dump next to :data:`DEFAULT_LOG`.
    """
    # Start each run from a clean log: truncate any prior transcript so the
    # file only ever holds the most recent sweep.
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")
    # Header block: pin the run's provenance (corpus, cadence, model set) so
    # the log is self-describing when read back later.
    log(f"# LLM model-size & family sweep — 2026-06-30")
    log(f"# corpus : {rttm}")
    log(f"# cadence : flush_every_s=60.0 (canonical winner)")
    log(f"# models : {MODELS}")

    # Load the transcript once and reuse it for every model — the utterances
    # are identical across candidates, only the LLM changes.
    utts = load_utterances(rttm)
    # Audio duration = span from first onset to last offset; it is the RTF
    # denominator, so compute it from the extreme timestamps, not len(utts).
    dur = max(u["t1"] for u in utts) - min(u["t0"] for u in utts)
    log(f"# utterances : {len(utts)}")
    log(f"# audio_duration : {dur:.1f}s")

    # rows : (model, ref_wall, candidate_wall, RTF, n_calls, cos_sim)
    rows: list[tuple[str, float, float, float, int, float]] = []

    # Sweep every candidate. Each model is judged against its OWN reference
    # (see module docstring) to strip out inter-model style noise.
    for model in MODELS:
        log(f"\n=== {model} ===")
        # Reference pass: one full-transcript call = this model's best shot.
        # If it fails, record a NaN row and move on rather than aborting the
        # whole sweep on one flaky backend.
        try:
            ref_text, ref_wall = reference_summary(utts, model, host)
        except Exception as exc:  # noqa: BLE001
            log(f"  reference FAILED : {exc!r}")
            rows.append((model, float("nan"), float("nan"), float("nan"), 0, 0.0))
            continue
        log(f"  reference   wall={ref_wall:6.1f}s  chars={len(ref_text)}")

        # Candidate pass: the rolling summary at the canonical cadence
        # (60 s flush, 60 s recent window). flush_every_n is set absurdly
        # high so the time-based flush is the only trigger that fires.
        try:
            text, wall, n_calls = await candidate_summary(
                utts,
                model=model,
                host=host,
                recent_window_s=60.0,
                flush_every_n=10_000,
                flush_every_s=60.0,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  candidate FAILED : {exc!r}")
            rows.append((model, ref_wall, float("nan"), float("nan"), 0, 0.0))
            continue

        # Quality = cosine similarity of candidate vs its own reference;
        # speed = wall / audio duration. Both land in the row for the table.
        sim = cosine_sim(text, ref_text)
        rtf = wall / dur
        rows.append((model, ref_wall, wall, rtf, n_calls, sim))
        log(f"  candidate   wall={wall:6.1f}s  RTF={rtf:.3f}  n={n_calls}  cos_sim={sim:.3f}")

    # ----- summary -----
    # Emit the headline result table, one row per model.
    log("\n" + "=" * 68)
    log("Summary")
    log("=" * 68)
    log(f"{'model':<20s}  {'ref_wall':>9s}  {'wall':>8s}  {'RTF':>7s}  {'n':>4s}  {'cos':>5s}")
    log("-" * 60)
    for model, ref_wall, wall, rtf, n, sim in rows:
        log(f"{model:<20s}  {ref_wall:>9.1f}  {wall:>8.1f}  {rtf:>7.3f}  {n:>4d}  {sim:>5.3f}")

    # Drop failed rows before ranking: r[3] is RTF, and NaN != NaN, so this
    # comparison keeps only the models that actually produced a number.
    feasible = [r for r in rows if r[3] == r[3]]  # NaN filter
    if feasible:
        # Single-axis winners: best quality (max cos_sim) and best speed
        # (min RTF) — the two extremes readers usually want called out.
        winner_quality = max(feasible, key=lambda r: r[5])
        winner_rtf = min(feasible, key=lambda r: r[3])
        log(
            f"\nBest cos_sim : {winner_quality[0]}  RTF={winner_quality[3]:.3f}  cos={winner_quality[5]:.3f}"
        )
        log(f"Best RTF     : {winner_rtf[0]}  RTF={winner_rtf[3]:.3f}  cos={winner_rtf[5]:.3f}")

        # Pareto frontier: the models a decision-maker should actually
        # choose between — no other config is both faster AND at least as
        # good (or as fast AND strictly better).
        log("\nPareto front (no other config dominates them) :")
        front = []
        for r in feasible:
            # r is dominated if some other config beats it on one axis
            # without losing on the other. If nothing dominates it, it is
            # on the frontier.
            dominated = any(
                (o[3] < r[3] and o[5] >= r[5]) or (o[3] <= r[3] and o[5] > r[5])
                for o in feasible
                if o != r
            )
            if not dominated:
                front.append(r)
                log(f"  {r[0]:<20s}  RTF={r[3]:.3f}  cos={r[5]:.3f}")

    # Machine-readable dump alongside the text log so downstream tooling can
    # reload the raw rows without re-parsing the printed table.
    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "corpus": str(rttm),
                "audio_duration_s": dur,
                "cadence": {"recent_window_s": 60.0, "flush_every_s": 60.0},
                "rows": [
                    {
                        "model": m,
                        "ref_wall_s": rw,
                        "wall_s": w,
                        "rtf": rtf,
                        "n_calls": n,
                        "cos_sim": sim,
                    }
                    for m, rw, w, rtf, n, sim in rows
                ],
            },
            indent=2,
        )
    )
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


# argparse used to hand the module docstring to --help ; it mangled the RST,
# so the rendered help is written for a terminal and the study write-up stays
# where it belongs, in the module docstring.
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--rttm",
    # argparse only coerced to Path here — it never checked the file exists,
    # so no ``exists=True``: a missing corpus still fails in load_utterances.
    type=click.Path(path_type=Path),
    default=DEFAULT_RTTM,
    show_default=True,
    help="Word-level RTTM supplying the transcript every model summarises.",
)
@click.option("--host", default=None, help="Ollama host URL (default localhost).")
def main(rttm: Path, host: str | None) -> None:
    """Sweep seven Ollama models for the analyst and draw the Pareto front.

    Each candidate is scored against its own one-shot reference summary
    (cosine similarity) and its RTF at the canonical 60 s cadence.

    \f

    Parameters
    ----------
    rttm : Path
        RTTM transcript to load utterances from.
    host : str or None
        Ollama host URL, or ``None`` for the default endpoint.

    Returns
    -------
    None
        Delegates all work to :func:`amain` via :func:`asyncio.run`.
    """
    # The ``\f`` closing the summary above is click's help-truncation marker:
    # click renders the *whole* docstring into ``--help``, so without it the
    # numpydoc blocks (and their RST double backticks) would reach users.
    # The sweep is async (candidate_summary streams the LLM); run it on a
    # fresh event loop.
    asyncio.run(amain(rttm=rttm, host=host))


if __name__ == "__main__":
    main()
