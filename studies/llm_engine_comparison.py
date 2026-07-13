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

DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/"
    "vocal_helper_llm_engine_2026-06-30.log"
)

QWEN_OLLAMA = "qwen2.5:3b"
QWEN_VLLM = "Qwen/Qwen2.5-3B-Instruct"
QWEN_MLX = "mlx-community/Qwen2.5-3B-Instruct-4bit"

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
NUM_TIMED = 5
NUM_WARMUP = 1


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(DEFAULT_LOG, "a") as f:
        f.write(msg + "\n")


def bench_ollama(host: str | None) -> dict:
    try:
        import ollama  # type: ignore
    except ImportError as e:
        return {"engine": "ollama", "status": f"missing : {e}"}
    client = ollama.Client(host=host) if host else ollama.Client()
    # Warm-up.
    try:
        client.generate(model=QWEN_OLLAMA, prompt=PROMPT, stream=False)
    except Exception as e:  # noqa: BLE001
        return {"engine": "ollama", "status": f"load failed : {e!r}"}
    walls: list[float] = []
    chars: list[int] = []
    for _ in range(NUM_TIMED):
        t0 = time.perf_counter()
        try:
            resp = client.generate(model=QWEN_OLLAMA, prompt=PROMPT, stream=False)
        except Exception as e:  # noqa: BLE001
            return {"engine": "ollama", "status": f"call failed : {e!r}"}
        walls.append(time.perf_counter() - t0)
        text = resp.get("response", "") if isinstance(resp, dict) else getattr(resp, "response", "")
        chars.append(len(str(text)))
    return {
        "engine": "ollama",
        "model": QWEN_OLLAMA,
        "status": "ok",
        "wall_min": min(walls),
        "wall_med": statistics.median(walls),
        "wall_max": max(walls),
        "chars_med": statistics.median(chars),
        "chars_per_s_med": statistics.median(chars) / statistics.median(walls),
    }


def bench_vllm() -> dict:
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except ImportError as e:
        return {"engine": "vllm", "status": f"missing : {e}"}
    try:
        llm = LLM(model=QWEN_VLLM, dtype="float16", gpu_memory_utilization=0.7)
    except Exception as e:  # noqa: BLE001
        return {"engine": "vllm", "status": f"load failed : {e!r}"}
    params = SamplingParams(max_tokens=400, temperature=0.7)
    # Warm-up.
    try:
        llm.generate([PROMPT], params)
    except Exception as e:  # noqa: BLE001
        return {"engine": "vllm", "status": f"warmup failed : {e!r}"}
    walls: list[float] = []
    chars: list[int] = []
    for _ in range(NUM_TIMED):
        t0 = time.perf_counter()
        out = llm.generate([PROMPT], params)
        walls.append(time.perf_counter() - t0)
        chars.append(len(out[0].outputs[0].text))
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
    try:
        from mlx_lm import load, generate  # type: ignore
    except ImportError as e:
        return {"engine": "mlx-lm", "status": f"missing : {e}"}
    try:
        model, tokenizer = load(QWEN_MLX)
    except Exception as e:  # noqa: BLE001
        return {"engine": "mlx-lm", "status": f"load failed : {e!r}"}
    # Warm-up.
    try:
        generate(model, tokenizer, prompt=PROMPT, max_tokens=400, verbose=False)
    except Exception as e:  # noqa: BLE001
        return {"engine": "mlx-lm", "status": f"warmup failed : {e!r}"}
    walls: list[float] = []
    chars: list[int] = []
    for _ in range(NUM_TIMED):
        t0 = time.perf_counter()
        out = generate(model, tokenizer, prompt=PROMPT, max_tokens=400, verbose=False)
        walls.append(time.perf_counter() - t0)
        chars.append(len(out))
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=None)
    args = p.parse_args()

    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")
    log("# LLM serving-engine comparison — 2026-06-30")
    log(f"# prompt size : {len(PROMPT)} chars")
    log(f"# n_timed     : {NUM_TIMED} calls + {NUM_WARMUP} warmup\n")

    results = []
    log("=== Ollama ===")
    r = bench_ollama(args.host); results.append(r)
    log(json.dumps(r, indent=2, default=str)); log("")

    log("=== vLLM ===")
    r = bench_vllm(); results.append(r)
    log(json.dumps(r, indent=2, default=str)); log("")

    log("=== mlx-lm ===")
    r = bench_mlx_lm(); results.append(r)
    log(json.dumps(r, indent=2, default=str)); log("")

    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        log("\n" + "=" * 56)
        log("Summary (median over runs)")
        log("=" * 56)
        log(f"{'engine':<10s}  {'wall_med':>10s}  {'chars/s':>10s}")
        log("-" * 32)
        for r in ok:
            log(f"{r['engine']:<10s}  {r['wall_med']:>10.2f}  {r['chars_per_s_med']:>10.1f}")
        # RTF projection
        log("\nProjected RTF on the canonical 869 s session (13 calls / session) :")
        for r in ok:
            llm_wall = r["wall_med"] * 13
            rtf = llm_wall / 869
            log(f"  {r['engine']:<10s}  llm_wall={llm_wall:>5.0f}s  RTF={rtf:.3f}")
        winner = min(ok, key=lambda r: r["wall_med"])
        log(f"\nFastest engine : {winner['engine']}")

    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


if __name__ == "__main__":
    main()
