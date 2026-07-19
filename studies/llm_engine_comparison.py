"""LLM serving-engine comparison — Ollama vs vLLM vs raw MLX-LM.

Question
--------
At identical model and identical prompt, does **vLLM** give a better
RTF than **Ollama** on the user's Apple-Silicon MacBook ?
And does direct **MLX-LM** (Apple's own inference stack) beat both ?

Caveats
-------
- vLLM is CUDA-first ; Apple-Silicon (MPS / Metal) support landed in
  v0.6 and is documented as experimental. The script catches import
  + load errors and reports them rather than crashing.
- MLX-LM (``mlx-lm`` pip package) runs MLX-converted weights natively
  on Apple-Silicon GPU. It's the lower-bound benchmark for "what
  fast can be" on this hardware.

Protocol
--------
1. Pick a representative summarization prompt of similar size to what
   ``GemmaAnalystStage`` actually sends (system prompt + previous
   summary + ~ 5 new utterances).
2. For each engine that successfully loads :
   - warm-up call (discarded, JIT compile / cache warmup) ;
   - 5 timed calls ;
   - record min / median / max wall-time.
3. Report :
   - wall-time per call ;
   - tokens / second (chars / second approximation if no token API) ;
   - RTF projection : how this would change the LLM stage's RTF on
     the canonical 869 s audio (assuming 13 calls per session per the
     winning cadence).

Models
------
Tested on Qwen 2.5 3B (gguf for Ollama, HF safetensors for vLLM,
MLX-converted for mlx-lm). Gemma 4 isn't trivially portable across
the three engines (different naming) so this study uses Qwen as the
common reference.

Author : Warith HARCHAOUI — 2026-06-30
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

# Where the human-readable run log (and, via ``.with_suffix``, the JSON dump)
# lands on the scratch volume.
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/vocal_helper_llm_engine_2026-06-30.log"
)

# Same Qwen 2.5 3B model expressed in each engine's own naming scheme, so the
# three benchmarks compare like-for-like weights.
QWEN_OLLAMA = "qwen2.5:3b"
QWEN_VLLM = "Qwen/Qwen2.5-3B-Instruct"
QWEN_MLX = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Fixed prompt sized like a real ``GemmaAnalystStage`` update (system + running
# summary + ~5 new utterances) so timings reflect production-shaped work.
PROMPT = (
    "You are a meeting note-taker. Update the running summary below "
    "by integrating the new utterances. Keep it concise (≤ 6 bullet "
    "points), preserve speaker attributions, and drop low-signal "
    "small talk. Output only the updated summary, nothing else.\n\n"
    "Current summary:\n"
    "- A introduced the team and outlined the remote-control project goal.\n"
    "- D explained the marketing plan is pending and depends on team alignment.\n"
    "- C clarified she has not begun industrial design pending project goal confirmation.\n\n"
    "New utterances (older → newer):\n"
    "[120.0-122.5] A: Let's make sure each of us has a deliverable for next week.\n"
    "[123.0-127.4] B: I'll send the UI mockups by Wednesday at the latest.\n"
    "[127.5-130.1] D: I need pricing from the supplier first.\n"
    "[131.0-135.2] C: I can begin the mechanical sketches as soon as we agree on form factor.\n"
    "[136.0-138.5] A: Great. Let's reconvene Friday with progress.\n"
)
# One warm-up call (absorbs JIT / model-load cost) then five timed calls whose
# median is reported.
NUM_TIMED = 5
NUM_WARMUP = 1


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
    # ``print`` is the documented study-result channel; the file copy keeps a
    # durable record of the benchmark run.
    print(msg, flush=True)
    with open(DEFAULT_LOG, "a") as f:
        f.write(msg + "\n")


def bench_ollama(host: str | None) -> dict:
    """Benchmark the Ollama engine on the shared summarisation prompt.

    Parameters
    ----------
    host : str | None
        Ollama host URL, or ``None`` to use the default localhost client.

    Returns
    -------
    dict
        A result record. On success it carries wall-time min/median/max, the
        median output length and chars/s; on any failure it carries only
        ``engine`` and a ``status`` string describing what went wrong, so the
        caller can report engines uniformly without try/except at the top level.
    """
    # Treat a missing optional dependency as a non-fatal "engine unavailable".
    try:
        import ollama  # type: ignore
    except ImportError as e:
        return {"engine": "ollama", "status": f"missing : {e}"}

    # Honour an explicit host; otherwise talk to the default localhost daemon.
    client = ollama.Client(host=host) if host else ollama.Client()

    # Warm-up: first call pays JIT / model-load / cache cost, so discard it.
    try:
        client.generate(model=QWEN_OLLAMA, prompt=PROMPT, stream=False)
    except Exception as e:  # noqa: BLE001
        return {"engine": "ollama", "status": f"load failed : {e!r}"}

    # Timed calls: collect wall-time and output length per iteration.
    walls: list[float] = []
    chars: list[int] = []
    for _ in range(NUM_TIMED):
        # Wall-clock around the single generate call is the RTF-relevant cost.
        t0 = time.perf_counter()
        try:
            resp = client.generate(model=QWEN_OLLAMA, prompt=PROMPT, stream=False)
        except Exception as e:  # noqa: BLE001
            return {"engine": "ollama", "status": f"call failed : {e!r}"}
        walls.append(time.perf_counter() - t0)
        # The response may be a dict or an object depending on client version;
        # pull ``response`` out of either shape.
        text = resp.get("response", "") if isinstance(resp, dict) else getattr(resp, "response", "")
        chars.append(len(str(text)))
    # Collapse the per-call samples into the record shape shared by all engines.
    return {
        "engine": "ollama",
        "model": QWEN_OLLAMA,
        "status": "ok",
        "wall_min": min(walls),
        "wall_med": statistics.median(walls),
        "wall_max": max(walls),
        "chars_med": statistics.median(chars),
        # chars/s from the medians — a token/s proxy since Ollama gives no
        # token count here.
        "chars_per_s_med": statistics.median(chars) / statistics.median(walls),
    }


def bench_vllm() -> dict:
    """Benchmark the vLLM engine on the shared summarisation prompt.

    Returns
    -------
    dict
        A result record with the same shape as :func:`bench_ollama` — wall-time
        stats + chars/s on success, or ``engine`` + ``status`` on failure.

    Notes
    -----
    vLLM is CUDA-first; on Apple-Silicon it is experimental and may fail to
    import or load. Those failures are caught and reported as ``status`` rather
    than raised, so the comparison still runs for the other engines.
    """
    # Optional dependency: absence just means "engine unavailable".
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except ImportError as e:
        return {"engine": "vllm", "status": f"missing : {e}"}

    # Model load can fail on unsupported hardware; report instead of crashing.
    try:
        llm = LLM(model=QWEN_VLLM, dtype="float16", gpu_memory_utilization=0.7)
    except Exception as e:  # noqa: BLE001
        return {"engine": "vllm", "status": f"load failed : {e!r}"}
    # Cap output length and match the other engines' sampling temperature.
    params = SamplingParams(max_tokens=400, temperature=0.7)

    # Warm-up: discard the first (compile / cache) call.
    try:
        llm.generate([PROMPT], params)
    except Exception as e:  # noqa: BLE001
        return {"engine": "vllm", "status": f"warmup failed : {e!r}"}

    # Timed calls: one wall-time and one output-length sample per iteration.
    walls: list[float] = []
    chars: list[int] = []
    for _ in range(NUM_TIMED):
        # Wall-clock around the single generate call is the RTF-relevant cost.
        t0 = time.perf_counter()
        out = llm.generate([PROMPT], params)
        walls.append(time.perf_counter() - t0)
        # vLLM returns a list of RequestOutput; the generated text is nested.
        chars.append(len(out[0].outputs[0].text))
    # Same record shape as the other engines so the summary loop is uniform.
    return {
        "engine": "vllm",
        "model": QWEN_VLLM,
        "status": "ok",
        "wall_min": min(walls),
        "wall_med": statistics.median(walls),
        "wall_max": max(walls),
        "chars_med": statistics.median(chars),
        "chars_per_s_med": statistics.median(chars) / statistics.median(walls),
    }


def bench_mlx_lm() -> dict:
    """Benchmark Apple's MLX-LM engine on the shared summarisation prompt.

    Returns
    -------
    dict
        A result record with the same shape as :func:`bench_ollama` — wall-time
        stats + chars/s on success, or ``engine`` + ``status`` on failure.

    Notes
    -----
    MLX-LM runs MLX-converted weights natively on the Apple-Silicon GPU, so it
    is the lower bound on "how fast this hardware can go" for the comparison.
    """
    # Optional dependency.
    try:
        from mlx_lm import generate, load  # type: ignore
    except ImportError as e:
        return {"engine": "mlx-lm", "status": f"missing : {e}"}

    # Load the converted weights + tokenizer; report a load failure inline.
    try:
        model, tokenizer = load(QWEN_MLX)
    except Exception as e:  # noqa: BLE001
        return {"engine": "mlx-lm", "status": f"load failed : {e!r}"}

    # Warm-up: discard the first (compile / cache) generation.
    try:
        generate(model, tokenizer, prompt=PROMPT, max_tokens=400, verbose=False)
    except Exception as e:  # noqa: BLE001
        return {"engine": "mlx-lm", "status": f"warmup failed : {e!r}"}

    # Timed calls: one wall-time and one output-length sample per iteration.
    walls: list[float] = []
    chars: list[int] = []
    for _ in range(NUM_TIMED):
        # Wall-clock around the single generate call is the RTF-relevant cost.
        t0 = time.perf_counter()
        out = generate(model, tokenizer, prompt=PROMPT, max_tokens=400, verbose=False)
        walls.append(time.perf_counter() - t0)
        # ``generate`` returns the decoded string directly here.
        chars.append(len(out))
    # Same record shape as the other engines so the summary loop is uniform.
    return {
        "engine": "mlx-lm",
        "model": QWEN_MLX,
        "status": "ok",
        "wall_min": min(walls),
        "wall_med": statistics.median(walls),
        "wall_max": max(walls),
        "chars_med": statistics.median(chars),
        "chars_per_s_med": statistics.median(chars) / statistics.median(walls),
    }


def main() -> None:
    """Benchmark all three engines and print the comparison + RTF projection.

    Returns
    -------
    None
        Results are emitted to stdout, appended to ``DEFAULT_LOG`` and dumped to
        a sibling ``.json`` file rather than returned.
    """
    # Only ``--host`` is configurable; models and prompt are pinned constants.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=None)
    args = p.parse_args()

    # Start from a clean log so each comparison run is self-contained.
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")
    # Header records the run's fixed parameters (prompt size, call counts).
    log("# LLM serving-engine comparison — 2026-06-30")
    log(f"# prompt size : {len(PROMPT)} chars")
    log(f"# n_timed     : {NUM_TIMED} calls + {NUM_WARMUP} warmup\n")

    # Run each engine in turn; each bench returns a self-describing record
    # (never raises) so one unavailable engine doesn't sink the run.
    results = []
    log("=== Ollama ===")
    r = bench_ollama(args.host)
    results.append(r)
    log(json.dumps(r, indent=2, default=str))
    log("")

    log("=== vLLM ===")
    r = bench_vllm()
    results.append(r)
    log(json.dumps(r, indent=2, default=str))
    log("")

    log("=== mlx-lm ===")
    r = bench_mlx_lm()
    results.append(r)
    log(json.dumps(r, indent=2, default=str))
    log("")

    # Only engines that actually ran feed the summary table and projection.
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        # Compact side-by-side table: median wall-time and chars/s per engine.
        log("\n" + "=" * 56)
        log("Summary (median over runs)")
        log("=" * 56)
        log(f"{'engine':<10s}  {'wall_med':>10s}  {'chars/s':>10s}")
        log("-" * 32)
        # One row per engine that produced timings.
        for r in ok:
            log(f"{r['engine']:<10s}  {r['wall_med']:>10.2f}  {r['chars_per_s_med']:>10.1f}")
        # RTF projection: extrapolate per-call wall to a full canonical session
        # (13 analyst calls over 869 s of audio) to compare engines in the
        # terms the pipeline actually cares about.
        log("\nProjected RTF on the canonical 869 s session (13 calls / session) :")
        for r in ok:
            # 13 calls × per-call median = total LLM wall over one session.
            llm_wall = r["wall_med"] * 13
            rtf = llm_wall / 869
            log(f"  {r['engine']:<10s}  llm_wall={llm_wall:>5.0f}s  RTF={rtf:.3f}")
        # Fastest = smallest median per-call wall-time.
        winner = min(ok, key=lambda r: r["wall_med"])
        log(f"\nFastest engine : {winner['engine']}")

    # Persist the raw records (including failures) for downstream inspection.
    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


if __name__ == "__main__":
    main()
