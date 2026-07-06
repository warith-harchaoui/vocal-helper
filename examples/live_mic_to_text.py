"""Live microphone → diarized transcript (+ optional Gemma summary).

Run with the default microphone, pyannote diarization, whisper.cpp
turbo and the Gemma 4 e4b analyst :

    pip install 'vocal-helper[all]'

Then supply the pyannote HuggingFace token by any of these means
(highest priority first) :

* pass ``--hf-token hf_…`` on the CLI ;
* export ``HF_TOKEN=hf_…`` in the shell ;
* copy ``settings.yaml.example`` to ``settings.yaml`` and fill in
  ``secrets.hf_token``.

Then run :

    python examples/live_mic_to_text.py --llm

Stop with Ctrl-C ; the pipeline cancels every running task cleanly.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

import vocal_helper as vh


async def amain(args: argparse.Namespace) -> None:
    # The ``hf_token=None`` here delegates resolution to the stage
    # constructor, which calls ``_settings.resolve_hf_token`` and walks
    # the documented order : env var, then settings.yaml.
    config = vh.PipelineConfig(
        diar={
            "backend": args.diar_backend,
            "hf_token": args.hf_token,
        },
        asr={
            "model": args.whisper_model,
            "language": args.language,
            "threads": args.threads,
        },
        llm=({"model": args.llm_model, "recent_window_s": 60.0} if args.llm else None),
    )
    pipeline = vh.Pipeline(
        source=lambda: vh.sources.from_microphone(device_name=args.device),
        config=config,
    )
    print("vocal-helper live — Ctrl-C to stop", file=sys.stderr)
    try:
        async for ev in pipeline.run():
            if "text" in ev:
                print(f"[{ev['t0']:7.2f}-{ev['t1']:7.2f} {ev['speaker']}] {ev['text']}")
            elif "summary" in ev:
                print(f"\n=== rolling summary @ {ev['t0']:.1f}s ===")
                print(ev["summary"])
                print("=== recent ===")
                print(ev["recent"])
                print("===\n")
    except KeyboardInterrupt:
        print("\nstopping…", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help="substring of the microphone name")
    p.add_argument("--whisper-model", default="large-v3-turbo-q5_0")
    p.add_argument("--language", default="auto")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--diar-backend", choices=["pyannote", "nemo"], default="pyannote")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (falls back to $HF_TOKEN, then "
                        "settings.yaml secrets.hf_token).")
    p.add_argument("--llm", action="store_true")
    p.add_argument("--llm-model", default="gemma3:4b")
    args = p.parse_args()
    asyncio.run(amain(args))
