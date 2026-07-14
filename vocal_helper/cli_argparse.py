"""
Vocal Helper — argparse-based command-line interface.

Thin wrapper around the async :class:`vocal_helper.Pipeline` /
:class:`vocal_helper.OfflinePipeline` orchestrators that exposes the
toolkit as subcommands under a single ``vocal-helper`` entry point.
Written with :mod:`argparse` from the standard library so the CLI
works out of the box on any Python install that has the package
installed — no extra dependency required.

Subcommands
-----------
- ``mic``        — live microphone input (needs ``[mic]`` extra)
- ``file``       — replay a mono 16 kHz WAV through the pipeline
- ``url``        — stream from any URL yt-dlp can reach (``[stream]`` extra)
- ``transcribe`` — one-shot transcription of a WAV / numpy buffer

Every subcommand accepts a common core of ``--whisper-model`` /
``--language`` / ``--diar-backend`` / ``--llm`` / ``--jsonl`` levers.
See ``vocal-helper <subcommand> --help`` for the full surface.

Usage Example
-------------
>>> #   vocal-helper mic --llm --jsonl
>>> #   vocal-helper file meeting.wav --offline --language en
>>> #   vocal-helper url "https://youtu.be/…" --language fr
>>> #   vocal-helper transcribe clip.wav --language en

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from vocal_helper.pipeline import (
    OfflinePipeline,
    OfflinePipelineConfig,
    Pipeline,
    PipelineConfig,
)

# ---------------------------------------------------------------------------
# Config builder — shared by every subcommand that spins up a pipeline.
#
# We keep the mapping "CLI namespace -> PipelineConfig" in one place so the
# click twin (:mod:`vocal_helper.cli_click`) and the FastAPI surface can
# reuse the exact same defaults without drift.
# ---------------------------------------------------------------------------


def _build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """
    Translate the parsed CLI namespace into a :class:`PipelineConfig`.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace with the common flags added by :func:`_add_common_flags`.

    Returns
    -------
    PipelineConfig
        A ready-to-use pipeline configuration (VAD / diar / ASR / LLM).
    """
    # ASR dict passed straight through to WhisperStage.__init__.
    # ``getattr`` guards partial Namespaces built by tests that predate a
    # given lever — the CLI always populates these, but the fallbacks keep
    # config building total.
    asr_cfg: dict = {
        "model": args.whisper_model,
        "language": args.language,
        "threads": args.threads,
        "initial_prompt": getattr(args, "initial_prompt", "") or "",
    }
    # Diar dict — pyannote or NeMo backend. Model weights load from the
    # self-hosted diarization-engines bundle (settings.yaml), no HF token.
    diar_cfg: dict = {"backend": args.diar_backend}
    if args.join_threshold is not None:
        diar_cfg["join_threshold"] = args.join_threshold
    # LLM stage is opt-in — omitting --llm leaves it disabled.
    llm_cfg: dict | None = None
    if args.llm:
        llm_cfg = {
            "model": args.llm_model,
            "recent_window_s": args.llm_recent_window_s,
        }
        if args.ollama_host:
            llm_cfg["host"] = args.ollama_host
    # SemanticEOTStage is opt-in — enabling it activates the LiveKit-style
    # turn detector that holds back VAD segments that look mid-thought. The
    # keys must match :class:`vocal_helper.eot.SemanticEOTStage.__init__`
    # (``eot_model`` / ``host``), since the pipeline splats this dict.
    eot_cfg: dict | None = None
    if getattr(args, "eot", False):
        eot_cfg = {}
        if getattr(args, "eot_model", None):
            eot_cfg["eot_model"] = args.eot_model
        if args.ollama_host:
            eot_cfg["host"] = args.ollama_host
    return PipelineConfig(diar=diar_cfg, asr=asr_cfg, llm=llm_cfg, eot=eot_cfg)


def _print_event(ev: dict, jsonl: bool) -> None:
    """Emit a single pipeline event to stdout in the requested format."""
    if jsonl:
        # Filter the raw PCM before serialising — a 20 ms buffer would blow
        # up log volume and JSON is not the right transport for float32.
        sys.stdout.write(json.dumps({k: v for k, v in ev.items() if k != "pcm"}) + "\n")
        sys.stdout.flush()
        return
    if "text" in ev:
        sys.stdout.write(f"[{ev['t0']:7.2f}s -> {ev['t1']:7.2f}s  {ev['speaker']}]  {ev['text']}\n")
    elif "summary" in ev:
        sys.stdout.write(
            f"\n--- rolling summary @ {ev['t0']:.1f}s "
            f"(model={ev['model']}) ---\n{ev['summary']}\n"
            f"--- recent window ---\n{ev['recent']}\n\n"
        )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Subcommand handlers — one per verb.
# ---------------------------------------------------------------------------


async def _run_pipeline(args: argparse.Namespace, source_factory) -> None:
    """Instantiate the right pipeline (online / offline) and drain events."""
    config = _build_pipeline_config(args)
    if getattr(args, "offline", False):
        # Offline pipeline skips the VAD and gives the diar backend the whole buffer.
        offline_cfg = OfflinePipelineConfig(
            diar=config.diar,
            asr=config.asr,
            llm=config.llm,
        )
        pipeline = OfflinePipeline(source=source_factory, config=offline_cfg)
    else:
        pipeline = Pipeline(source=source_factory, config=config)
    async for ev in pipeline.run():
        _print_event(ev, args.jsonl)


def _handle_mic(args: argparse.Namespace) -> int:
    # Import lazily so users without the [mic] extra can still use file/url.
    from vocal_helper.sources import from_microphone

    def factory():
        return from_microphone(
            device_name=args.device,
            sample_rate=16_000,
            frame_ms=20,
        )

    asyncio.run(_run_pipeline(args, factory))
    return 0


def _handle_file(args: argparse.Namespace) -> int:
    from vocal_helper.sources import from_wav_file

    path = Path(args.path)

    def factory():
        return from_wav_file(path, real_time=not args.no_real_time)

    asyncio.run(_run_pipeline(args, factory))
    return 0


def _handle_url(args: argparse.Namespace) -> int:
    # URL streaming needs podcast_helper (the [stream] extra).
    from vocal_helper.sources import from_url

    def factory():
        return from_url(args.url)

    asyncio.run(_run_pipeline(args, factory))
    return 0


def _handle_transcribe(args: argparse.Namespace) -> int:
    """One-shot synchronous transcription of a WAV file."""
    # Lazy imports so ``vocal-helper --help`` never pays the numpy /
    # audio-helper / whisper.cpp cost for people who only wanted usage text.
    import numpy as np
    from audio_helper import load_audio

    from vocal_helper.asr import transcribe_pcm

    # ffmpeg-backed decode — any format (mp3/m4a/opus/video), mono, native rate.
    pcm, sr = load_audio(args.path, to_mono=True, to_numpy=True)
    pcm = np.asarray(pcm, dtype=np.float32)
    text = transcribe_pcm(
        pcm=pcm,
        sr=int(sr),
        model=args.whisper_model,
        language=args.language,
        threads=args.threads,
        initial_prompt=getattr(args, "initial_prompt", "") or "",
    )
    if args.jsonl:
        sys.stdout.write(json.dumps({"path": args.path, "text": text}) + "\n")
    else:
        sys.stdout.write(text + "\n")
    sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# Parser construction — one helper per subcommand keeps ``build_parser``
# readable and lets the click twin mirror the exact same flag names
# without drift.
# ---------------------------------------------------------------------------


def _add_common_flags(sp: argparse.ArgumentParser) -> None:
    """Attach the shared VAD / diar / ASR / LLM levers to a subparser."""
    sp.add_argument(
        "--whisper-model",
        default="large-v3-turbo-q5_0",
        help="pywhispercpp model tag (default large-v3-turbo-q5_0).",
    )
    sp.add_argument(
        "--language", default="auto", help="ISO-639-1 code (en/fr/…) or 'auto' for language ID."
    )
    sp.add_argument("--threads", type=int, default=6, help="whisper.cpp CPU threads (default 6).")
    sp.add_argument(
        "--initial-prompt",
        default="",
        help="Whisper bias prompt — name the conversational domain "
        "and a few expected proper nouns / technical terms. "
        "Strongly recommended: cuts WER 15-25 pp and saves up to "
        "39%% RTF on AMI (2026-06-30 sweep). Example: 'medical "
        "telemedicine consultation: patient symptoms, medication "
        "review, follow-up appointment'.",
    )
    sp.add_argument(
        "--diar-backend",
        choices=["pyannote", "nemo"],
        default="nemo",
        help="Speaker-embedding backend for the online diarizer. "
        "Default 'nemo' (TitaNet) — +76%% separability margin over "
        "'pyannote' on AMI (2026-06-30 sweep). Switch to 'pyannote' "
        "to skip the ~5 GB NeMo install.",
    )
    sp.add_argument(
        "--join-threshold",
        type=float,
        default=None,
        help="Cosine-distance join threshold for the online diarizer (default 0.30).",
    )
    sp.add_argument(
        "--llm", action="store_true", help="Enable the Gemma analyst stage (rolling summary)."
    )
    sp.add_argument(
        "--llm-model",
        default="gemma3:4b",
        help="Ollama model tag (default gemma3:4b — Pareto sweet spot of "
        "the 2026-06-30 7-model sweep).",
    )
    sp.add_argument(
        "--llm-recent-window-s",
        type=float,
        default=60.0,
        help="Verbatim window (seconds) kept out of the summary (default 60).",
    )
    sp.add_argument(
        "--ollama-host",
        default=None,
        help="Override for the Ollama server host (default 127.0.0.1:11434).",
    )
    sp.add_argument(
        "--eot",
        action="store_true",
        help="Enable the SemanticEOTStage (LiveKit-style turn detector). "
        "Holds back VAD segments that look mid-thought and merges "
        "them with their successor, reducing mid-sentence cuts at the "
        "cost of one extra LLM hop per voiced segment.",
    )
    sp.add_argument(
        "--eot-model",
        default=None,
        help="Ollama model for the EOT completeness classifier (default qwen2.5:3b).",
    )
    sp.add_argument(
        "--jsonl",
        action="store_true",
        help="Emit one JSON event per line instead of human-readable output.",
    )


def _add_mic(sub: argparse._SubParsersAction) -> None:
    """Live microphone input."""
    p = sub.add_parser("mic", help="Live microphone input (needs the [mic] extra).")
    _add_common_flags(p)
    p.add_argument(
        "--device",
        default=None,
        help="Substring of the microphone name to select a specific device.",
    )
    p.set_defaults(func=_handle_mic)


def _add_file(sub: argparse._SubParsersAction) -> None:
    """Replay a WAV file through the pipeline."""
    p = sub.add_parser("file", help="Replay a 16 kHz mono WAV through the pipeline.")
    _add_common_flags(p)
    p.add_argument("path", type=str, help="Path to the WAV file to process.")
    p.add_argument(
        "--no-real-time",
        action="store_true",
        help="Process as fast as possible (skip real-time pacing).",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use the OfflinePipeline (pyannote 3.1 full-buffer + "
        "auto chunk+stitch past 300 s) — highest quality on "
        "meetings, podcasts, lectures.",
    )
    p.set_defaults(func=_handle_file)


def _add_url(sub: argparse._SubParsersAction) -> None:
    """Stream from any URL yt-dlp can reach."""
    p = sub.add_parser(
        "url",
        help="Stream from any URL yt-dlp can reach (needs the [stream] extra).",
    )
    _add_common_flags(p)
    p.add_argument("url", type=str, help="YouTube / Vimeo / podcast RSS / direct audio URL.")
    p.set_defaults(func=_handle_url)


def _add_transcribe(sub: argparse._SubParsersAction) -> None:
    """One-shot ASR of a WAV file — no VAD, no diarization."""
    p = sub.add_parser(
        "transcribe",
        help="One-shot transcription of a WAV file (skip VAD / diarization).",
    )
    p.add_argument("path", type=str, help="Path to a WAV file.")
    p.add_argument("--whisper-model", default="large-v3-turbo-q5_0")
    p.add_argument("--language", default="auto")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument(
        "--initial-prompt",
        default="",
        help="Whisper bias prompt — name the domain and a few expected "
        "proper nouns. Cuts WER 15-25 pp on AMI (2026-06-30 sweep).",
    )
    p.add_argument(
        "--jsonl", action="store_true", help='Emit {"path": ..., "text": ...} JSON on stdout.'
    )
    p.set_defaults(func=_handle_transcribe)


def build_parser() -> argparse.ArgumentParser:
    """
    Assemble the top-level ``vocal-helper`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Fully wired parser with every subcommand attached.
    """
    parser = argparse.ArgumentParser(
        prog="vocal-helper",
        description=(
            "Vocal Helper — async producer/consumer pipeline turning audio "
            "into diarized, transcribed utterances (and, optionally, a rolling "
            "LLM summary). Subcommands: mic / file / url / transcribe."
        ),
    )
    # ``--version`` is cheap to add and oncall people always look for it.
    try:
        from importlib.metadata import version as _pkg_version

        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {_pkg_version('vocal-helper')}",
        )
    except Exception:  # pragma: no cover — never fatal
        pass

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    _add_mic(subparsers)
    _add_file(subparsers)
    _add_url(subparsers)
    _add_transcribe(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point invoked by ``vocal-helper`` (see ``[project.scripts]``).

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. Defaults to ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
