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
import statistics
import time
from collections import Counter
from pathlib import Path

from vocal_helper.llm import GemmaAnalystStage, _extract_response_text
from vocal_helper.types import Utterance

DEFAULT_RTTM = Path(
    "/Users/warithharchaoui/pasdebonneoudemauvaisesituation/"
    "data/ami/dev-slice/IS1008a/words.rttm"
)
DEFAULT_MODEL = "gemma4:e4b-mlx"  # MLX variant — Apple-Silicon-friendly
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/"
    "vocal_helper_llm_cadence_2026-06-30.log"
)
DEFAULT_BRIDGE_S = 0.3   # join adjacent same-speaker words within this gap


def load_utterances(rttm: Path) -> list[Utterance]:
    """Parse a word-level RTTM into bridged turn-level :class:`Utterance`s."""
    words: list[tuple[float, float, str, str]] = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        if len(p) < 10 or p[0] != "SPEAKER":
            continue
        t0 = float(p[3])
        dur = float(p[4])
        spk = p[7]
        word = p[-1]
        words.append((t0, t0 + dur, spk, word))
    words.sort(key=lambda x: x[0])
    out: list[Utterance] = []
    cur_t0 = cur_t1 = 0.0
    cur_spk: str | None = None
    cur_tokens: list[str] = []
    for t0, t1, spk, w in words:
        if cur_spk == spk and (t0 - cur_t1) <= DEFAULT_BRIDGE_S:
            cur_t1 = max(cur_t1, t1)
            cur_tokens.append(w)
        else:
            if cur_spk is not None and cur_tokens:
                out.append(Utterance(
                    t0=cur_t0, t1=cur_t1,
                    speaker=cur_spk, text=" ".join(cur_tokens),
                    words=[], language="en",
                ))
            cur_t0, cur_t1, cur_spk = t0, t1, spk
            cur_tokens = [w]
    if cur_spk is not None and cur_tokens:
        out.append(Utterance(
            t0=cur_t0, t1=cur_t1,
            speaker=cur_spk, text=" ".join(cur_tokens),
            words=[], language="en",
        ))
    return out


def transcript_to_text(utts: list[Utterance]) -> str:
    return "\n".join(
        f"[{u['t0']:.1f}-{u['t1']:.1f}] {u['speaker']}: {u['text']}"
        for u in utts
    )


# ---------------------------------------------------------------------------
# Quality metric — TF-IDF cosine sim. Deterministic, no extra LLM call.
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")


def _tokens(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


def _tf_idf(docs: list[str]) -> list[Counter]:
    """Return per-doc TF-IDF Counter vectors (unweighted IDF on log)."""
    import math

    tokenised = [_tokens(d) for d in docs]
    df: Counter = Counter()
    for toks in tokenised:
        for t in set(toks):
            df[t] += 1
    idf = {t: math.log(1 + len(docs) / df[t]) for t in df}
    vecs = []
    for toks in tokenised:
        tf = Counter(toks)
        vecs.append(Counter({t: tf[t] * idf[t] for t in tf}))
    return vecs


def _cosine(a: Counter, b: Counter) -> float:
    import math

    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def cosine_sim(a: str, b: str) -> float:
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
    """Run a single LLM call on the full transcript ; return (summary, wall_s)."""
    import ollama  # type: ignore

    client = ollama.Client(host=host) if host else ollama.Client()
    transcript = transcript_to_text(utts)
    prompt = _REF_PROMPT.format(transcript=transcript)
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
    """Replay ``utts`` through the analyst ; return (final_summary, wall_s, n_calls)."""
    stage = GemmaAnalystStage(
        model=model,
        recent_window_s=recent_window_s,
        flush_every_n=flush_every_n,
        flush_every_s=flush_every_s,
        host=host,
    )
    # Patch ``_summarise`` to record wall-time and call count.
    state = {"wall_s": 0.0, "calls": 0}
    original = stage._summarise

    def _timed():
        t0 = time.perf_counter()
        out = original()
        state["wall_s"] += time.perf_counter() - t0
        state["calls"] += 1
        return out

    stage._summarise = _timed
    stage._ensure_client()
    inbox: asyncio.Queue = asyncio.Queue()
    outbox: asyncio.Queue = asyncio.Queue()
    for u in utts:
        await inbox.put(u)
    await inbox.put(None)
    task = asyncio.create_task(stage.run(inbox, outbox))
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
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(DEFAULT_LOG, "a") as f:
            f.write(msg + "\n")

    log(f"# LLM cadence sweep — 2026-06-30")
    log(f"# corpus : {args.rttm}")
    log(f"# model  : {args.model}")
    utts = load_utterances(args.rttm)
    duration_s = max(u["t1"] for u in utts) - min(u["t0"] for u in utts)
    log(f"# utterances : {len(utts)}")
    log(f"# audio_duration : {duration_s:.1f}s")

    # ----- reference -----
    log("\n[reference] one-shot summary on the full transcript …")
    ref_text, ref_wall = reference_summary(utts, args.model, args.host)
    log(f"  reference  wall={ref_wall:6.1f}s   chars={len(ref_text)}")
    log(f"  reference RTF (single shot) = {ref_wall / duration_s:.3f}")
    log("\n--- reference summary ---")
    log(ref_text)
    log("--- end reference ---")

    # ----- sweep -----
    configs: list[tuple[str, int, float | None]] = [
        ("n=1",  1,  None),
        ("n=2",  2,  None),
        ("n=3",  3,  None),
        ("n=5",  5,  None),
        ("n=10", 10, None),
        ("n=20", 20, None),
        ("t=30s",  10_000, 30.0),
        ("t=60s",  10_000, 60.0),
        ("t=120s", 10_000, 120.0),
    ]
    log("\n[sweep]")
    log(f"{'config':<10s}  {'wall_s':>8s}  {'RTF':>7s}  {'n_calls':>8s}  "
        f"{'cos_sim':>8s}")
    rows = []
    for label, fn, fs in configs:
        log(f"\n  running {label} …")
        text, wall, n = await candidate_summary(
            utts,
            model=args.model, host=args.host,
            recent_window_s=60.0,
            flush_every_n=fn, flush_every_s=fs,
        )
        sim = cosine_sim(text, ref_text)
        rtf = wall / duration_s
        rows.append((label, wall, rtf, n, sim, text))
        log(f"  {label:<10s}  {wall:>8.1f}  {rtf:>7.3f}  {n:>8d}  {sim:>8.3f}")

    # ----- summary table -----
    log("\n" + "=" * 60)
    log("Summary table")
    log("=" * 60)
    log(f"{'config':<10s}  {'wall_s':>8s}  {'RTF':>7s}  {'n_calls':>8s}  "
        f"{'cos_sim':>8s}")
    log("-" * 60)
    for label, wall, rtf, n, sim, _ in rows:
        log(f"{label:<10s}  {wall:>8.1f}  {rtf:>7.3f}  {n:>8d}  {sim:>8.3f}")

    # ----- Pareto pick : highest cos_sim with RTF ≤ 0.10 -----
    feasible = [r for r in rows if r[2] <= 0.10]
    if not feasible:
        feasible = rows
    winner = max(feasible, key=lambda r: r[4])
    log(f"\nWinner (highest cos_sim with RTF ≤ 0.10) : {winner[0]}")
    log(f"  wall={winner[1]:.1f}s  RTF={winner[2]:.3f}  "
        f"n_calls={winner[3]}  cos_sim={winner[4]:.3f}")
    log("\n--- winner final summary ---")
    log(winner[5])
    log("--- end winner ---")

    # JSON dump so downstream can pick up the result.
    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(json.dumps({
        "model": args.model,
        "audio_duration_s": duration_s,
        "reference_wall_s": ref_wall,
        "reference_rtf": ref_wall / duration_s,
        "winner": winner[0],
        "configs": [
            {
                "label": label, "wall_s": wall, "rtf": rtf,
                "n_calls": n, "cos_sim": sim, "summary": text,
            }
            for label, wall, rtf, n, sim, text in rows
        ],
    }, indent=2))
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rttm", type=Path, default=DEFAULT_RTTM)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=None,
                   help="Ollama host URL (default localhost)")
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
