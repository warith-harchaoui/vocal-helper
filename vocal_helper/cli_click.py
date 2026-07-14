"""
Vocal Helper — click-based command-line interface.

Twin of :mod:`vocal_helper.cli_argparse`: same public surface (identical
subcommand names, identical flag semantics), but implemented with
:mod:`click` so users who already have a click-native shell setup
(bash / zsh completion via ``click.shell_completion``, colored ``--help``,
nested command groups) can plug it in without friction. Installed as
the ``vocal-helper-click`` entry point in ``pyproject.toml``.

Design notes
------------
- Subcommands mirror ``vocal-helper`` (the argparse twin) so both CLIs
  can be introspected identically by higher layers (FastAPI, MCP).
- Flags reuse the argparse names (``--whisper-model``, ``--language``,
  ``--diar-backend``, …) rather than the more idiomatic click positional
  style — consistency across the two CLIs beats micro-idiomaticity.
- Async work is wrapped in :func:`asyncio.run` inside each command;
  click itself stays sync.

Usage Example
-------------
>>> #   vocal-helper-click mic --llm --jsonl
>>> #   vocal-helper-click file meeting.wav --offline --language en
>>> #   vocal-helper-click url "https://youtu.be/…" --language fr
>>> #   vocal-helper-click transcribe clip.wav --language en

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

try:
    import click
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The click CLI requires the [cli] extra. Install with: pip install 'vocal-helper[cli]'"
    ) from exc

from vocal_helper.pipeline import (
    OfflinePipeline,
    OfflinePipelineConfig,
    Pipeline,
    PipelineConfig,
)

# ---------------------------------------------------------------------------
# Shared config translation — mirrors the argparse twin's ``_build_pipeline_config``.
# We keep the click callbacks small by punching the shared kwargs through
# a plain dict rather than a Namespace.
# ---------------------------------------------------------------------------


def _pipeline_config(
    *,
    whisper_model: str,
    language: str,
    threads: int,
    initial_prompt: str,
    diar_backend: str,
    join_threshold: float | None,
    llm: bool,
    llm_model: str,
    llm_recent_window_s: float,
    ollama_host: str | None,
    eot: bool,
    eot_model: str | None,
) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from the shared click options."""
    # ASR dict passed straight through to WhisperStage.__init__ — keys mirror
    # the argparse twin so both CLIs produce byte-identical pipeline configs.
    asr_cfg: dict = {
        "model": whisper_model,
        "language": language,
        "threads": threads,
        # Coerce a possible ``None`` bias prompt to "" — whisper rejects None.
        "initial_prompt": initial_prompt or "",
    }
    # Model weights load from the self-hosted diarization-engines bundle
    # (settings.yaml ``engines.diarization_url``) — no HuggingFace token.
    diar_cfg: dict = {"backend": diar_backend}
    if join_threshold is not None:
        diar_cfg["join_threshold"] = join_threshold
    # LLM analyst is opt-in — leave it None unless ``--llm`` was passed.
    llm_cfg: dict | None = None
    if llm:
        llm_cfg = {"model": llm_model, "recent_window_s": llm_recent_window_s}
        # Only forward the host override when the user set one ; otherwise the
        # stage falls back to $OLLAMA_HOST / the localhost default itself.
        if ollama_host:
            llm_cfg["host"] = ollama_host
    # Keys must match SemanticEOTStage.__init__ (``eot_model`` / ``host``).
    eot_cfg: dict | None = None
    if eot:
        eot_cfg = {}
        if eot_model:
            eot_cfg["eot_model"] = eot_model
        if ollama_host:
            eot_cfg["host"] = ollama_host
    return PipelineConfig(diar=diar_cfg, asr=asr_cfg, llm=llm_cfg, eot=eot_cfg)


def _print_event(ev: dict, jsonl: bool) -> None:
    """Emit a single pipeline event to stdout, JSONL or human-readable."""
    if jsonl:
        # Strip the raw PCM before serialising — cheaper log, right transport.
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


async def _drain(pipeline, jsonl: bool) -> None:
    """Consume every event emitted by ``pipeline.run()``."""
    async for ev in pipeline.run():
        _print_event(ev, jsonl)


# ---------------------------------------------------------------------------
# Reusable option bundle — Click does not have a built-in shared-options
# decorator, but we can compose one with a helper. Every subcommand slaps
# ``@_common_options`` on top of its own signature.
# ---------------------------------------------------------------------------


def _common_options(func):
    """Decorator adding the shared VAD / diar / ASR / LLM levers."""
    # Order matters for --help output; we mirror the argparse twin.
    func = click.option(
        "--jsonl", is_flag=True, default=False, help="Emit one JSON event per line."
    )(func)
    func = click.option(
        "--eot-model",
        default=None,
        help="Ollama model for the EOT completeness classifier (default qwen2.5:3b).",
    )(func)
    func = click.option(
        "--eot",
        is_flag=True,
        default=False,
        help="Enable the SemanticEOTStage (LiveKit-style turn "
        "detector). Reduces mid-sentence cuts at the cost of "
        "one extra LLM hop per voiced segment.",
    )(func)
    func = click.option("--ollama-host", default=None, help="Override for the Ollama server host.")(
        func
    )
    func = click.option(
        "--llm-recent-window-s",
        type=float,
        default=60.0,
        show_default=True,
        help="Verbatim window (seconds) kept out of the summary.",
    )(func)
    func = click.option(
        "--llm-model",
        default="gemma3:4b",
        show_default=True,
        help="Ollama model tag (Pareto sweet spot of the 2026-06-30 sweep).",
    )(func)
    func = click.option(
        "--llm", is_flag=True, default=False, help="Enable the Gemma analyst stage."
    )(func)
    func = click.option(
        "--join-threshold",
        type=float,
        default=None,
        help="Cosine-distance join threshold for online diarizer (default 0.30).",
    )(func)
    func = click.option(
        "--diar-backend",
        type=click.Choice(["pyannote", "nemo"]),
        default="nemo",
        show_default=True,
        help="Speaker-embedding backend for the online diarizer. "
        "'nemo' (TitaNet) is default; 'pyannote' skips the "
        "~5 GB NeMo install.",
    )(func)
    func = click.option(
        "--initial-prompt",
        default="",
        help="Whisper bias prompt — name the domain and a few expected "
        "proper nouns. Cuts WER 15-25 pp on AMI (2026-06-30 sweep).",
    )(func)
    func = click.option(
        "--threads", type=int, default=6, show_default=True, help="whisper.cpp CPU threads."
    )(func)
    func = click.option(
        "--language",
        default="auto",
        show_default=True,
        help="ISO-639-1 code or 'auto' for language ID.",
    )(func)
    func = click.option(
        "--whisper-model",
        default="large-v3-turbo-q5_0",
        show_default=True,
        help="pywhispercpp model tag.",
    )(func)
    return func


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.version_option(package_name="vocal-helper", prog_name="vocal-helper-click")
def cli() -> None:
    """Vocal Helper — click twin of the argparse CLI. Same subcommands."""
    # Nothing at the group level; every subcommand carries its own args.


# ---------------------------------------------------------------------------
# mic
# ---------------------------------------------------------------------------


@cli.command()
@_common_options
@click.option("--device", default=None, help="Substring of the microphone name.")
def mic(
    whisper_model: str,
    language: str,
    threads: int,
    initial_prompt: str,
    diar_backend: str,
    join_threshold: float | None,
    llm: bool,
    llm_model: str,
    llm_recent_window_s: float,
    ollama_host: str | None,
    eot: bool,
    eot_model: str | None,
    jsonl: bool,
    device: str | None,
) -> None:
    """Live microphone input (needs the ``[mic]`` extra)."""
    from vocal_helper.sources import from_microphone

    cfg = _pipeline_config(
        whisper_model=whisper_model,
        language=language,
        threads=threads,
        initial_prompt=initial_prompt,
        diar_backend=diar_backend,
        join_threshold=join_threshold,
        llm=llm,
        llm_model=llm_model,
        llm_recent_window_s=llm_recent_window_s,
        ollama_host=ollama_host,
        eot=eot,
        eot_model=eot_model,
    )

    def factory():
        """Open a fresh 16 kHz / 20 ms microphone stream on each pipeline start."""
        # 16 kHz mono is whisper.cpp's native rate ; 20 ms is the Silero VAD stride,
        # so the source hands the pipeline frames it can consume without resampling.
        return from_microphone(device_name=device, sample_rate=16_000, frame_ms=20)

    pipeline = Pipeline(source=factory, config=cfg)
    asyncio.run(_drain(pipeline, jsonl))


# ---------------------------------------------------------------------------
# file
# ---------------------------------------------------------------------------


@cli.command()
@_common_options
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--no-real-time",
    is_flag=True,
    default=False,
    help="Process as fast as possible (skip real-time pacing).",
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help="Use OfflinePipeline for highest-quality diarization on long inputs.",
)
def file(
    whisper_model: str,
    language: str,
    threads: int,
    initial_prompt: str,
    diar_backend: str,
    join_threshold: float | None,
    llm: bool,
    llm_model: str,
    llm_recent_window_s: float,
    ollama_host: str | None,
    eot: bool,
    eot_model: str | None,
    jsonl: bool,
    path: str,
    no_real_time: bool,
    offline: bool,
) -> None:
    """Replay a 16 kHz mono WAV through the pipeline."""
    from vocal_helper.sources import from_wav_file

    cfg = _pipeline_config(
        whisper_model=whisper_model,
        language=language,
        threads=threads,
        initial_prompt=initial_prompt,
        diar_backend=diar_backend,
        join_threshold=join_threshold,
        llm=llm,
        llm_model=llm_model,
        llm_recent_window_s=llm_recent_window_s,
        ollama_host=ollama_host,
        eot=eot,
        eot_model=eot_model,
    )

    def factory():
        """Open the WAV source ; ``--no-real-time`` skips wall-clock pacing for batch runs."""
        # Real-time pacing simulates a live feed ; disabling it fires frames as fast
        # as they decode, which is what you want when timing throughput on a file.
        return from_wav_file(Path(path), real_time=not no_real_time)

    # ``--offline`` swaps in the full-buffer diarizer (best quality on long inputs) ;
    # otherwise we run the streaming pipeline that diarizes segment-by-segment.
    if offline:
        pipeline = OfflinePipeline(
            source=factory,
            config=OfflinePipelineConfig(diar=cfg.diar, asr=cfg.asr, llm=cfg.llm),
        )
    else:
        pipeline = Pipeline(source=factory, config=cfg)
    asyncio.run(_drain(pipeline, jsonl))


# ---------------------------------------------------------------------------
# url
# ---------------------------------------------------------------------------


@cli.command()
@_common_options
@click.argument("url")
def url(
    whisper_model: str,
    language: str,
    threads: int,
    initial_prompt: str,
    diar_backend: str,
    join_threshold: float | None,
    llm: bool,
    llm_model: str,
    llm_recent_window_s: float,
    ollama_host: str | None,
    eot: bool,
    eot_model: str | None,
    jsonl: bool,
    url: str,
) -> None:
    """Stream from any URL yt-dlp can reach (needs the ``[stream]`` extra)."""
    from vocal_helper.sources import from_url as _from_url

    cfg = _pipeline_config(
        whisper_model=whisper_model,
        language=language,
        threads=threads,
        initial_prompt=initial_prompt,
        diar_backend=diar_backend,
        join_threshold=join_threshold,
        llm=llm,
        llm_model=llm_model,
        llm_recent_window_s=llm_recent_window_s,
        ollama_host=ollama_host,
        eot=eot,
        eot_model=eot_model,
    )

    def factory():
        """Open a streaming source for the given URL (yt-dlp resolves the media)."""
        return _from_url(url)

    pipeline = Pipeline(source=factory, config=cfg)
    asyncio.run(_drain(pipeline, jsonl))


# ---------------------------------------------------------------------------
# transcribe — one-shot, no VAD/diarization.
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--whisper-model", default="large-v3-turbo-q5_0", show_default=True)
@click.option("--language", default="auto", show_default=True)
@click.option("--threads", type=int, default=6, show_default=True)
@click.option(
    "--initial-prompt",
    default="",
    help="Whisper bias prompt — name the domain and a few expected proper "
    "nouns. Cuts WER 15-25 pp on AMI (2026-06-30 sweep).",
)
@click.option(
    "--jsonl", is_flag=True, default=False, help='Emit {"path": ..., "text": ...} JSON on stdout.'
)
def transcribe(
    path: str,
    whisper_model: str,
    language: str,
    threads: int,
    initial_prompt: str,
    jsonl: bool,
) -> None:
    """One-shot transcription of a WAV file (skip VAD / diarization)."""
    # Lazy imports so ``--help`` never pays the numpy / audio-helper / whisper.cpp
    # import cost for users who only wanted the usage text.
    import numpy as np
    from audio_helper import load_audio

    from vocal_helper.asr import transcribe_pcm

    # ffmpeg-backed decode — any format (mp3/m4a/opus/video), mono, native rate.
    pcm, sr = load_audio(path, to_mono=True, to_numpy=True)
    # whisper.cpp wants a contiguous float32 buffer — coerce whatever dtype we got.
    pcm = np.asarray(pcm, dtype=np.float32)
    text = transcribe_pcm(
        pcm=pcm,
        sr=int(sr),
        model=whisper_model,
        language=language,
        threads=threads,
        initial_prompt=initial_prompt or "",
    )
    if jsonl:
        click.echo(json.dumps({"path": path, "text": text}))
    else:
        click.echo(text)


if __name__ == "__main__":  # pragma: no cover
    cli()
