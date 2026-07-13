"""Live URL (YouTube / Twitch / RSS / direct audio) → diarized transcript.

Streams any ``yt-dlp``-supported URL through the same online pipeline
as ``live_mic_to_text.py`` — VAD → online diarization → Whisper turbo →
(optionally) Gemma 4 e4b rolling summary.

Install :

    pip install 'vocal-helper[all]'

(``[all]`` brings the ``stream`` extra, which pulls ``podcast-helper`` —
that one wraps ``yt-dlp`` + ``ffmpeg`` for URL ingestion.)

Then run a YouTube VOD at real-time pace :

    python examples/live_url_to_text.py \\
        "https://www.youtube.com/watch?v=YE7VzlLtp-4" --llm

…or burst-transcribe a VOD at 2× speed (offline benchmark) :

    python examples/live_url_to_text.py \\
        "https://www.youtube.com/watch?v=YE7VzlLtp-4" \\
        --no-realtime --speed 2.0

Twitch live works the same way (``--no-realtime`` and ``--speed`` are
both ignored on live streams ; the source paces itself).

Stop with Ctrl-C ; the pipeline cancels every running task cleanly.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import vocal_helper as vh


async def amain(args: argparse.Namespace) -> None:
    # Same shape as live_mic_to_text — only the source factory changes.
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
        source=lambda: vh.sources.from_url(
            args.url,
            realtime=args.realtime,
            speed=args.speed,
            cookies_from_browser=args.cookies_from_browser,
            record_to=args.record_to,
        ),
        config=config,
    )
    print(f"vocal-helper streaming {args.url} — Ctrl-C to stop", file=sys.stderr)
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
    p.add_argument("url", help="YouTube / Twitch / RSS / direct audio URL")
    p.add_argument("--whisper-model", default="large-v3-turbo-q5_0")
    p.add_argument("--language", default="auto")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--diar-backend", choices=["pyannote", "nemo"], default="pyannote")
    p.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (falls back to $HF_TOKEN, then settings.yaml secrets.hf_token).",
    )
    p.add_argument("--llm", action="store_true")
    p.add_argument("--llm-model", default="gemma4:e4b")
    # Realtime is the live-like default. --no-realtime is for VOD batches.
    p.add_argument("--realtime", dest="realtime", action="store_true", default=True)
    p.add_argument("--no-realtime", dest="realtime", action="store_false")
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="VOD playback rate (atempo). Raises on live streams.",
    )
    p.add_argument(
        "--cookies-from-browser",
        default=None,
        help="firefox / chrome / safari — used by yt-dlp for age-gated or private content.",
    )
    p.add_argument(
        "--record-to",
        default=None,
        help="Optional path for a parallel compressed archive "
        "(podcast-helper's record_to passthrough).",
    )
    args = p.parse_args()
    asyncio.run(amain(args))
