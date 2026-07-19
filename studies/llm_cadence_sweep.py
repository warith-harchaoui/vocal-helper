"""LLM rolling-summary cadence sweep.

Goal
----
Find the operating point of :class:`vocal_helper.llm.GemmaAnalystStage`
that maximises :

- **Quality** : how close the streaming-built summary is to the
  *offline reference* summary (one LLM call on the entire transcript).
- **RTF** : sum of LLM wall-time / total audio duration. Smaller is
  better — RTF ≪ 1 means the analyst comfortably keeps up with the
  conversation in real time.

These two criteria pull in opposite directions :

- Many small flushes → high quality (every recent transition gets folded
  in), but more LLM calls → higher RTF.
- One end-of-conversation flush → minimal LLM cost (~ RTF for one call),
  but the summary is stale during the meeting.

Sweep parameters
----------------
- ``recent_window_s`` is held at **60 s** (user spec : "résumé glissant
  jusqu'à 1 minute avant le temps courant").
- ``flush_every_n`` ∈ {1, 2, 3, 5, 10, 20}.
- Plus one time-based config : ``flush_every_s = 60``.

Corpus
------
AMI dev-slice IS1008a — 16 min, ~ 4 speakers, real meeting cadence.
The reference utterances are pulled from ``words.rttm`` (the same
word-level RTTM used by the diar study), bridged at 200 ms to form
turn-level utterances.

Reference summary
-----------------
We treat the **single LLM call on the full transcript** as the
oracle. Quality of a candidate = TF-IDF cosine similarity between
its final summary and the reference summary.

Run
---
::

    python studies/llm_cadence_sweep.py

Expects Ollama running locally with the configured model available.

Author : Warith HARCHAOUI — 2026-06-30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path

from vocal_helper.llm import GemmaAnalystStage, _extract_response_text
from vocal_helper.types import Utterance

DEFAULT_RTTM = Path(
    "/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice/IS1008a/words.rttm"
)
DEFAULT_MODEL = "gemma4:e4b-mlx"  # MLX variant — Apple-Silicon-friendly
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/vocal_helper_llm_cadence_2026-06-30.log"
)
DEFAULT_BRIDGE_S = 0.3  # join adjacent same-speaker words within this gap


def load_utterances(rttm: Path) -> list[Utterance]:
    """Parse a word-level RTTM into bridged turn-level :class:`Utterance`s.

    Parameters
    ----------
    rttm : Path
        Path to a word-level RTTM file. Each ``SPEAKER`` line is expected
        to carry a start time (field 3), duration (field 4), a speaker
        label (field 7) and the transcribed word as its last field.

    Returns
    -------
    list[Utterance]
        Turn-level utterances built by merging consecutive same-speaker
        words whose inter-word gap does not exceed ``DEFAULT_BRIDGE_S``.

    Notes
    -----
    The word-level granularity of the RTTM is deliberately collapsed into
    turns so the transcript fed to the LLM reads like real speaker turns
    rather than one bullet per word.
    """
    # First pass: pull every word row out of the RTTM as a flat tuple of
    # (start, end, speaker, word). Anything that is not a well-formed
    # SPEAKER line is skipped so partial / malformed files don't crash us.
    words: list[tuple[float, float, str, str]] = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        # RTTM SPEAKER rows have at least 10 whitespace-separated fields;
        # skip headers, comments and any short/foreign line.
        if len(p) < 10 or p[0] != "SPEAKER":
            continue
        t0 = float(p[3])  # onset in seconds
        dur = float(p[4])  # duration in seconds
        spk = p[7]  # speaker label
        word = p[-1]  # the transcribed token itself
        words.append((t0, t0 + dur, spk, word))

    # RTTM rows are not guaranteed to be time-ordered; sort by onset so the
    # bridging pass below sees words in the order they were actually spoken.
    words.sort(key=lambda x: x[0])

    # Second pass: bridge adjacent same-speaker words into turn-level
    # utterances. ``cur_*`` accumulates the turn currently being built.
    out: list[Utterance] = []
    cur_t0 = cur_t1 = 0.0
    cur_spk: str | None = None
    cur_tokens: list[str] = []
    for t0, t1, spk, w in words:
        # Same speaker and a short enough gap → this word continues the
        # current turn, so extend its span and append the token.
        if cur_spk == spk and (t0 - cur_t1) <= DEFAULT_BRIDGE_S:
            cur_t1 = max(cur_t1, t1)
            cur_tokens.append(w)
        else:
            # Speaker changed (or the gap is too large): flush the turn we
            # were building before starting a fresh one.
            if cur_spk is not None and cur_tokens:
                out.append(
                    Utterance(
                        t0=cur_t0,
                        t1=cur_t1,
                        speaker=cur_spk,
                        text=" ".join(cur_tokens),
                        words=[],
                        language="en",
                    )
                )
            # Open a new turn seeded with the current word.
            cur_t0, cur_t1, cur_spk = t0, t1, spk
            cur_tokens = [w]

    # Flush the final in-progress turn — the loop above only emits a turn
    # when the *next* one starts, so the last turn is never emitted inside it.
    if cur_spk is not None and cur_tokens:
        out.append(
            Utterance(
                t0=cur_t0,
                t1=cur_t1,
                speaker=cur_spk,
                text=" ".join(cur_tokens),
                words=[],
                language="en",
            )
        )
    return out


def transcript_to_text(utts: list[Utterance]) -> str:
    """Render utterances as a plain-text transcript for the LLM prompt.

    Parameters
    ----------
    utts : list[Utterance]
        Turn-level utterances to serialise.

    Returns
    -------
    str
        One line per utterance, formatted ``[t0-t1] speaker: text``, so the
        model sees timing and speaker attribution alongside the words.
    """
    # One line per turn, carrying the timestamps and speaker so the model can
    # preserve attributions in its summary.
    return "\n".join(f"[{u['t0']:.1f}-{u['t1']:.1f}] {u['speaker']}: {u['text']}" for u in utts)


# ---------------------------------------------------------------------------
# Quality metric — TF-IDF cosine sim. Deterministic, no extra LLM call.
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")


def _tokens(s: str) -> list[str]:
    """Lower-case word tokens (≥ 2 letters) extracted from a string.

    Parameters
    ----------
    s : str
        Arbitrary text to tokenise.

    Returns
    -------
    list[str]
        Lower-cased alphabetic tokens; digits and punctuation are dropped so
        the TF-IDF vectors compare content words rather than numbers.
    """
    # ``_TOKEN_RE`` already excludes 1-char tokens and non-letters; just
    # lower-case what it finds so casing doesn't split otherwise-equal words.
    return [t.lower() for t in _TOKEN_RE.findall(s)]


def _tf_idf(docs: list[str]) -> list[Counter]:
    """Return per-doc TF-IDF Counter vectors (unweighted IDF on log).

    Parameters
    ----------
    docs : list[str]
        Corpus of documents to vectorise. IDF is computed over exactly this
        list, so the vectors are only comparable within one call.

    Returns
    -------
    list[Counter]
        One sparse term → weight Counter per input document, aligned by index
        with ``docs``.
    """
    import math

    # Tokenise every document once; reused for both DF and TF below.
    tokenised = [_tokens(d) for d in docs]

    # Document frequency: count in how many docs each term appears (use a set
    # so repeats within a single doc don't inflate DF).
    df: Counter = Counter()
    for toks in tokenised:
        for t in set(toks):
            df[t] += 1

    # Smoothed IDF: log(1 + N/df) keeps weights positive and dampens very
    # common terms without ever going negative.
    idf = {t: math.log(1 + len(docs) / df[t]) for t in df}

    # Combine term frequency with IDF to get the final per-doc weight vectors.
    vecs = []
    for toks in tokenised:
        tf = Counter(toks)
        vecs.append(Counter({t: tf[t] * idf[t] for t in tf}))
    return vecs


def _cosine(a: Counter, b: Counter) -> float:
    """Cosine similarity between two sparse weight vectors.

    Parameters
    ----------
    a, b : Counter
        Sparse term → weight vectors (as produced by :func:`_tf_idf`).

    Returns
    -------
    float
        Cosine similarity in ``[0, 1]``; ``0.0`` when either vector is empty
        or has zero norm (i.e. no shared or no non-zero terms).
    """
    import math

    # An empty vector has no direction, so similarity is undefined → treat as 0.
    if not a or not b:
        return 0.0

    # Only terms present in both vectors contribute to the dot product; the
    # rest multiply by zero.
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)

    # Norms over the full vectors, not just the shared terms.
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))

    # Guard against a zero norm (all-zero weights) before dividing.
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def cosine_sim(a: str, b: str) -> float:
    """TF-IDF cosine similarity between two raw text strings.

    Parameters
    ----------
    a, b : str
        The two texts to compare (typically a candidate summary and the
        offline reference summary).

    Returns
    -------
    float
        Cosine similarity in ``[0, 1]``; higher means the candidate summary is
        closer to the reference.
    """
    # Build the two TF-IDF vectors over just this pair, then compare them.
    va, vb = _tf_idf([a, b])
    return _cosine(va, vb)


# ---------------------------------------------------------------------------
# Reference summary — one LLM call on the full transcript.
# ---------------------------------------------------------------------------


_REF_PROMPT = (
    "You are a meeting note-taker. Summarise the following transcript "
    "into a concise digest (≤ 6 bullet points), preserve speaker "
    "attributions, drop low-signal small talk. Output only the digest.\n\n"
    "Transcript:\n{transcript}\n"
)


def reference_summary(utts: list[Utterance], model: str, host: str | None) -> tuple[str, float]:
    """Run a single LLM call on the full transcript ; return (summary, wall_s).

    Parameters
    ----------
    utts : list[Utterance]
        The full meeting transcript, already turn-bridged.
    model : str
        Ollama model tag to summarise with.
    host : str | None
        Ollama host URL, or ``None`` to use the default localhost client.

    Returns
    -------
    tuple[str, float]
        ``(summary_text, wall_seconds)`` — the oracle summary and how long the
        single call took, used both as the quality reference and to report the
        one-shot RTF.

    Notes
    -----
    This one-shot digest is treated as the *oracle*: every candidate cadence is
    scored by how close its streaming summary lands to this text.
    """
    import ollama  # type: ignore

    # Honour an explicit host when given; otherwise talk to the default
    # localhost Ollama daemon.
    client = ollama.Client(host=host) if host else ollama.Client()

    # Serialise the whole conversation and fold it into the reference prompt.
    transcript = transcript_to_text(utts)
    prompt = _REF_PROMPT.format(transcript=transcript)

    # Time only the generation call so the reported wall / RTF is comparable to
    # the streaming candidates below.
    t0 = time.perf_counter()
    resp = client.generate(model=model, prompt=prompt, stream=False)
    wall = time.perf_counter() - t0
    return _extract_response_text(resp), wall


# ---------------------------------------------------------------------------
# Candidate — stream the transcript through GemmaAnalystStage.
# ---------------------------------------------------------------------------


async def candidate_summary(
    utts: list[Utterance],
    *,
    model: str,
    host: str | None,
    recent_window_s: float,
    flush_every_n: int,
    flush_every_s: float | None,
) -> tuple[str, float, int]:
    """Replay ``utts`` through the analyst ; return (final_summary, wall_s, n_calls).

    Parameters
    ----------
    utts : list[Utterance]
        The meeting transcript to stream through the analyst, in order.
    model : str
        Ollama model tag the analyst stage should use.
    host : str | None
        Ollama host URL, or ``None`` for the default localhost client.
    recent_window_s : float
        Width of the rolling "recent" window (seconds) the analyst folds in.
    flush_every_n : int
        Trigger a summarisation every N utterances.
    flush_every_s : float | None
        Trigger a summarisation every this-many seconds, or ``None`` to rely
        solely on the N-based trigger.

    Returns
    -------
    tuple[str, float, int]
        ``(final_summary, wall_seconds, n_calls)`` — the last summary emitted,
        the cumulative LLM wall-time (only the summarise calls, not queue
        plumbing), and how many summarise calls the cadence produced.

    Notes
    -----
    ``stage._summarise`` is monkey-patched with a timing wrapper so we can
    attribute wall-time and call count to the LLM alone, isolating it from the
    async queue overhead of the replay harness.
    """
    stage = GemmaAnalystStage(
        model=model,
        recent_window_s=recent_window_s,
        flush_every_n=flush_every_n,
        flush_every_s=flush_every_s,
        host=host,
    )
    # Patch ``_summarise`` to record wall-time and call count. ``state`` is a
    # mutable cell the closure below writes into (nonlocal-by-dict).
    state = {"wall_s": 0.0, "calls": 0}
    original = stage._summarise

    def _timed():
        """Timing wrapper around the stage's real ``_summarise``.

        Returns
        -------
        object
            Whatever the wrapped ``_summarise`` returns, unchanged; the wrapper
            only accumulates wall-time and the call count in ``state`` as a
            side effect.
        """
        # Measure just the LLM call so queue/plumbing time is excluded.
        t0 = time.perf_counter()
        out = original()
        state["wall_s"] += time.perf_counter() - t0
        state["calls"] += 1
        return out

    # Install the timing wrapper and eagerly create the client so client
    # construction doesn't leak into the first timed call.
    stage._summarise = _timed
    stage._ensure_client()

    # Drive the stage over asyncio queues, exactly as the live pipeline would.
    inbox: asyncio.Queue = asyncio.Queue()
    outbox: asyncio.Queue = asyncio.Queue()
    for u in utts:
        await inbox.put(u)
    # A sentinel ``None`` tells the stage the stream is finished.
    await inbox.put(None)
    task = asyncio.create_task(stage.run(inbox, outbox))

    # Consume the outbox, keeping the most recent non-empty summary as the
    # final one; stop on the ``None`` end-of-stream sentinel.
    final_summary = ""
    while True:
        item = await outbox.get()
        if item is None:
            break
        final_summary = item["summary"] or final_summary
    await task
    return final_summary, state["wall_s"], state["calls"]


# ---------------------------------------------------------------------------
# Main sweep.
# ---------------------------------------------------------------------------


async def amain(args: argparse.Namespace) -> None:
    """Run the full single-meeting cadence sweep and write the result table.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments carrying ``rttm`` (corpus path), ``model`` (Ollama
        model tag) and ``host`` (optional Ollama host URL).

    Returns
    -------
    None
        Results are emitted to stdout, appended to ``DEFAULT_LOG`` and dumped
        as a sibling ``.json`` file rather than returned.
    """
    # Make sure the log directory exists and start from an empty log so each
    # run's output is self-contained.
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
        # ``print`` is the documented study-result channel here (see module
        # docstring / task note); the file copy keeps a durable record.
        print(msg, flush=True)
        with open(DEFAULT_LOG, "a") as f:
            f.write(msg + "\n")

    # Header block: pin down which corpus and model this run used.
    log("# LLM cadence sweep — 2026-06-30")
    log(f"# corpus : {args.rttm}")
    log(f"# model  : {args.model}")

    # Load the transcript and derive the audio span (used as the RTF denominator).
    utts = load_utterances(args.rttm)
    duration_s = max(u["t1"] for u in utts) - min(u["t0"] for u in utts)
    log(f"# utterances : {len(utts)}")
    log(f"# audio_duration : {duration_s:.1f}s")

    # ----- reference -----
    # Build the oracle summary first; every candidate is scored against it.
    log("\n[reference] one-shot summary on the full transcript …")
    ref_text, ref_wall = reference_summary(utts, args.model, args.host)
    log(f"  reference  wall={ref_wall:6.1f}s   chars={len(ref_text)}")
    log(f"  reference RTF (single shot) = {ref_wall / duration_s:.3f}")
    log("\n--- reference summary ---")
    log(ref_text)
    log("--- end reference ---")

    # ----- sweep -----
    # Cadence grid: N-based flushes at increasing intervals, then a few
    # time-based flushes. ``(label, flush_every_n, flush_every_s)``.
    configs: list[tuple[str, int, float | None]] = [
        ("n=1", 1, None),
        ("n=2", 2, None),
        ("n=3", 3, None),
        ("n=5", 5, None),
        ("n=10", 10, None),
        ("n=20", 20, None),
        ("t=30s", 10_000, 30.0),
        ("t=60s", 10_000, 60.0),
        ("t=120s", 10_000, 120.0),
    ]
    log("\n[sweep]")
    log(f"{'config':<10s}  {'wall_s':>8s}  {'RTF':>7s}  {'n_calls':>8s}  {'cos_sim':>8s}")
    # Run each cadence, scoring its final summary against the reference and
    # collecting a row per config for the summary table below.
    rows = []
    for label, fn, fs in configs:
        log(f"\n  running {label} …")
        # ``recent_window_s`` is fixed at 60 s per the user spec; only the
        # flush cadence varies across the sweep.
        text, wall, n = await candidate_summary(
            utts,
            model=args.model,
            host=args.host,
            recent_window_s=60.0,
            flush_every_n=fn,
            flush_every_s=fs,
        )
        # Quality = closeness to the oracle; RTF = LLM wall over audio span.
        sim = cosine_sim(text, ref_text)
        rtf = wall / duration_s
        rows.append((label, wall, rtf, n, sim, text))
        log(f"  {label:<10s}  {wall:>8.1f}  {rtf:>7.3f}  {n:>8d}  {sim:>8.3f}")

    # ----- summary table -----
    log("\n" + "=" * 60)
    log("Summary table")
    log("=" * 60)
    log(f"{'config':<10s}  {'wall_s':>8s}  {'RTF':>7s}  {'n_calls':>8s}  {'cos_sim':>8s}")
    log("-" * 60)
    # Re-print each row compactly (drop the summary text) for the final table.
    for label, wall, rtf, n, sim, _ in rows:
        log(f"{label:<10s}  {wall:>8.1f}  {rtf:>7.3f}  {n:>8d}  {sim:>8.3f}")

    # ----- Pareto pick : highest cos_sim with RTF ≤ 0.10 -----
    # Keep only real-time-feasible configs (RTF within budget), then pick the
    # one whose summary is closest to the oracle.
    feasible = [r for r in rows if r[2] <= 0.10]
    # If nothing meets the RTF budget, fall back to ranking all configs so the
    # sweep still names a best-effort winner.
    if not feasible:
        feasible = rows
    winner = max(feasible, key=lambda r: r[4])
    log(f"\nWinner (highest cos_sim with RTF ≤ 0.10) : {winner[0]}")
    log(
        f"  wall={winner[1]:.1f}s  RTF={winner[2]:.3f}  "
        f"n_calls={winner[3]}  cos_sim={winner[4]:.3f}"
    )
    log("\n--- winner final summary ---")
    log(winner[5])
    log("--- end winner ---")

    # JSON dump so downstream can pick up the result.
    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "model": args.model,
                "audio_duration_s": duration_s,
                "reference_wall_s": ref_wall,
                "reference_rtf": ref_wall / duration_s,
                "winner": winner[0],
                "configs": [
                    {
                        "label": label,
                        "wall_s": wall,
                        "rtf": rtf,
                        "n_calls": n,
                        "cos_sim": sim,
                        "summary": text,
                    }
                    for label, wall, rtf, n, sim, text in rows
                ],
            },
            indent=2,
        )
    )
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


def main() -> None:
    """Parse CLI arguments and run the single-meeting cadence sweep.

    Returns
    -------
    None
        Delegates to :func:`amain` under the asyncio event loop; all output is
        side-effect (stdout / log / JSON).
    """
    # Reuse the module docstring as the CLI help text so ``--help`` explains the
    # study, not just the flags.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rttm", type=Path, default=DEFAULT_RTTM)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=None, help="Ollama host URL (default localhost)")
    args = p.parse_args()
    # The sweep is async (queue-driven analyst); run it to completion here.
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
