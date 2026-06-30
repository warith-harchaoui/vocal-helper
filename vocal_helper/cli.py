"""
vocal_helper.cli
================

``vocal-helper`` — minimal CLI to wire the pipeline end-to-end from
the shell.

Two subcommands :

- ``vocal-helper mic`` — live microphone input, prints utterances
  (and summaries if ``--llm`` is given) to stdout.
- ``vocal-helper file PATH.wav`` — replay a mono 16 kHz WAV through
  the same pipeline.

Both modes accept ``--language``, ``--whisper-model``, ``--diar-backend``,
``--llm`` / ``--llm-model`` and a few other levers ; see ``--help`` for
the full surface.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from vocal_helper._settings import resolve_hf_token
from vocal_helper.pipeline import (
    OfflinePipeline,
    OfflinePipelineConfig,
    Pipeline,
    PipelineConfig,
)
from vocal_helper.sources import from_microphone, from_wav_file


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    asr_cfg: dict = {
        "model": args.whisper_model,
        "language": args.language,
        "threads": args.threads,
    }
    diar_cfg: dict = {"backend": args.diar_backend}
    # Resolution order : --hf-token > $HF_TOKEN > settings.yaml secrets.hf_token.
    token = resolve_hf_token(args.hf_token)
    if token:
        diar_cfg["hf_token"] = token
    if args.join_threshold is not None:
        diar_cfg["join_threshold"] = args.join_threshold
    llm_cfg: dict | None = None
    if args.llm:
        llm_cfg = {
            "model": args.llm_model,
            "recent_window_s": args.llm_recent_window_s,
        }
        if args.ollama_host:
            llm_cfg["host"] = args.ollama_host
    return PipelineConfig(diar=diar_cfg, asr=asr_cfg, llm=llm_cfg)


async def _amain(args: argparse.Namespace) -> None:
    config = _build_config(args)
    if args.cmd == "mic":
        source = lambda: from_microphone(  # noqa: E731 — closure for the factory
            device_name=args.device, sample_rate=16_000, frame_ms=20,
        )
    elif args.cmd == "file":
        path = Path(args.path)
        source = lambda: from_wav_file(  # noqa: E731
            path, real_time=not args.no_real_time,
        )
    else:
        raise SystemExit(f"unknown subcommand {args.cmd!r}")

    if getattr(args, "offline", False):
        # Offline pipeline doesn't use the VAD ; the diar backend
        # consumes the full buffer.
        offline_cfg = OfflinePipelineConfig(
            diar=config.diar,
            asr=config.asr,
            llm=config.llm,
        )
        pipeline = OfflinePipeline(source=source, config=offline_cfg)
    else:
        pipeline = Pipeline(source=source, config=config)
    async for ev in pipeline.run():
        if args.jsonl:
            sys.stdout.write(json.dumps({k: v for k, v in ev.items() if k != "pcm"}) + "\n")
            sys.stdout.flush()
        else:
            if "text" in ev:
                sys.stdout.write(
                    f"[{ev['t0']:7.2f}s → {ev['t1']:7.2f}s  {ev['speaker']}]  {ev['text']}\n"
                )
            elif "summary" in ev:
                sys.stdout.write(
                    f"\n--- rolling summary @ {ev['t0']:.1f}s "
                    f"(model={ev['model']}) ---\n{ev['summary']}\n"
                    f"--- recent window ---\n{ev['recent']}\n\n"
                )
            sys.stdout.flush()


def main() -> None:
    p = argparse.ArgumentParser(prog="vocal-helper", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--whisper-model", default="large-v3-turbo-q5_0")
        sp.add_argument("--language", default="auto")
        sp.add_argument("--threads", type=int, default=6)
        sp.add_argument("--diar-backend", choices=["pyannote", "nemo"], default="pyannote")
        sp.add_argument("--hf-token", default=None,
                        help="HuggingFace token for pyannote model fetch. "
                             "Falls back to $HF_TOKEN then settings.yaml "
                             "(secrets.hf_token) when omitted.")
        sp.add_argument("--join-threshold", type=float, default=None,
                        help="cosine-distance join threshold for the online diarizer (default 0.30)")
        sp.add_argument("--llm", action="store_true",
                        help="enable the Gemma analyst stage")
        sp.add_argument("--llm-model", default="gemma4:e4b")
        sp.add_argument("--llm-recent-window-s", type=float, default=60.0)
        sp.add_argument("--ollama-host", default=None)
        sp.add_argument("--jsonl", action="store_true",
                        help="emit one JSON event per line instead of human output")

    mic = sub.add_parser("mic", help="live microphone input")
    add_common(mic)
    mic.add_argument("--device", default=None, help="substring of the microphone name")

    f = sub.add_parser("file", help="replay a 16 kHz mono WAV through the pipeline")
    add_common(f)
    f.add_argument("path", type=str)
    f.add_argument("--no-real-time", action="store_true",
                   help="process as fast as possible (skip real-time pacing)")
    f.add_argument("--offline", action="store_true",
                   help="use the OfflinePipeline (pyannote 3.1 full-buffer + "
                        "auto chunk+stitch past 300 s) — highest quality on "
                        "meetings, podcasts, lectures, voicemail batches")

    args = p.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
