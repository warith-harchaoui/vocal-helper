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
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from vocal_helper.pipeline import (
    OfflinePipeline,
    OfflinePipelineConfig,
    Pipeline,
    PipelineConfig,
    SourceFactory,
)
from vocal_helper.router import select_diarization
from vocal_helper.types import PcmFrame

# ---------------------------------------------------------------------------
# Config builder — shared by every subcommand that spins up a pipeline.
#
# We keep the mapping "CLI namespace -> PipelineConfig" in one place so the
# click twin (:mod:`vocal_helper.cli_click`) can reuse the exact same
# defaults without drift.
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


def _print_event(ev: Mapping[str, Any], jsonl: bool) -> None:
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


def _offline_pyannote_available() -> bool:
    """True when the reliable offline pyannote path can run without a download.

    Cheap, side-effect-free pre-flight (no model load): the ``pyannote`` extra
    must import and the self-hosted diarization-engines bundle must already
    carry the ``pyannote-3.1`` config locally. Used to decide whether a batch
    file run can auto-upgrade to the offline diarizer.
    """
    try:
        import pyannote.audio  # type: ignore # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return False
    try:
        from vocal_helper.diar import resolve_diarization_engines

        engines = resolve_diarization_engines()
    except Exception:  # noqa: BLE001
        return False
    return (
        engines is not None
        and (engines / "pyannote-3.1" / "pyannote_diarization_config.yaml").exists()
    )


def _offline_nemo_available() -> bool:
    """True when the NeMo Sortformer backend can actually run (extra importable).

    Cheap import probe (no model load): the ``nemo`` extra pulls ``torch`` +
    ``nemo_toolkit``. The router must not route a short file to NeMo when the
    extra is absent, so this gates its short/dense branch — otherwise the plan
    would name an unrunnable backend and the offline stage would crash.
    """
    try:
        import nemo.collections.asr  # type: ignore # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return False
    return True


def _route_backend(
    *,
    requested_backend: str,
    live: bool,
    duration_s: float | None,
) -> tuple[str, str | None]:
    """Resolve the diarization backend for one run — router-decided or overridden.

    The single choke-point that turns ``--diar-backend`` into a concrete backend
    so the *aiguilleur* is actually enforced (never decorative). ``"auto"`` (the
    default) hands the decision to :func:`~vocal_helper.router.select_diarization`
    with the run's real conditions — live-vs-batch, probed ``duration_s``, and
    which backends are installed — reporting both **quality (DER)** and **speed
    (RTF)**. Any explicit backend is honoured as an operator override.

    Parameters
    ----------
    requested_backend : str
        ``"auto"`` to route, or ``"pyannote"`` / ``"nemo"`` / ``"sherpa"`` to
        override.
    live : bool
        ``True`` for a live stream, ``False`` for a batch file.
    duration_s : float or None
        Probed audio length in seconds (``None`` = unknown ⇒ long-form branch).

    Returns
    -------
    tuple of (str, str or None)
        The concrete backend name and a one-line stderr note (the router's
        quality/speed rationale, or an override notice), or ``None`` for no note.
    """
    # An explicit backend is the operator's call — honour it, but surface a note
    # so an online pyannote/nemo override (which the study says loses) is visible.
    if requested_backend != "auto":
        return requested_backend, None
    # "auto" → the router decides from the real scenario conditions.
    plan = select_diarization(
        live=live,
        duration_s=duration_s,
        torch_free=False,
        pyannote_available=_offline_pyannote_available(),
        nemo_available=_offline_nemo_available(),
    )
    note = (
        f"vocal-helper: router → {plan.mode} {plan.backend} "
        f"(quality DER {plan.expected_der}, speed RTF {plan.expected_rtf}; {plan.reason}). "
        "Pass --diar-backend to override."
    )
    return plan.backend, note


def _choose_file_diar(
    base_diar: dict,
    *,
    explicit_offline: bool,
    batch: bool,
    force_online: bool,
    duration_s: float | None = None,
    requested_backend: str = "auto",
) -> tuple[bool, dict, str | None]:
    """Pick online-vs-offline **and** the backend for a batch file run.

    Shared by the argparse and click CLIs so the decision can't drift between
    them, and the one place the study-grounded router (the *aiguilleur*) is
    enforced for files. The offline whole-buffer path is the reliability default
    for batch (2026-07-16 DER sweep: offline 0.12 / 0.34 vs online 0.50 / 0.59);
    *which* offline backend runs is the router's call, now fed the file's real
    probed ``duration_s`` so the short→nemo / long→pyannote crossover actually
    fires. Precedence:

    - explicit ``--offline`` → offline; router picks the backend from the real
      duration (or the caller's ``--diar-backend`` override).
    - batch, not ``--online``, an offline backend is installed → offline, backend
      routed by duration; falls back to the online diarizer only when no offline
      backend can run.
    - ``--online`` / non-batch → online streaming diarizer (backend routed too),
      with ``refine_on_close`` for batch so long audio doesn't over-segment.

    Returns ``(use_offline, diar_cfg, note)`` with ``diar_cfg`` always carrying a
    concrete backend (never ``"auto"``) and ``note`` a one-line stderr message
    (or ``None``).
    """
    pyannote_ok = _offline_pyannote_available()
    nemo_ok = _offline_nemo_available()
    explicit = requested_backend != "auto"
    # A file the operator forced offline: route the backend by real duration
    # (honouring an explicit --diar-backend), and run the whole-buffer path.
    if explicit_offline:
        backend, note = _route_backend(
            requested_backend=requested_backend, live=False, duration_s=duration_s
        )
        return True, {**base_diar, "backend": backend}, note
    # Auto path: prefer offline for batch when an offline backend can actually
    # run, and let the router choose it from the probed length.
    offline_runnable = pyannote_ok or nemo_ok or (explicit and requested_backend == "sherpa")
    if batch and not force_online and offline_runnable:
        backend, note = _route_backend(
            requested_backend=requested_backend, live=False, duration_s=duration_s
        )
        return True, {**base_diar, "backend": backend}, note
    # Batch but no offline backend installed → online streaming with the global
    # refine pass (the runnable fallback); still route the online backend.
    if batch:
        backend, _ = _route_backend(
            requested_backend=requested_backend, live=True, duration_s=duration_s
        )
        note = (
            None
            if force_online
            else (
                "vocal-helper: no offline diarizer installed — using the online "
                "diarizer with the refine pass. For best reliability install "
                "vocal-helper[pyannote] and configure the diarization-engines "
                "bundle (settings.yaml), then re-run."
            )
        )
        return False, {**base_diar, "backend": backend, "refine_on_close": True}, note
    # Non-batch file replay (real-time pacing) → online, backend routed.
    backend, note = _route_backend(
        requested_backend=requested_backend, live=True, duration_s=duration_s
    )
    return False, {**base_diar, "backend": backend}, note


async def _run_pipeline(args: argparse.Namespace, source_factory: SourceFactory) -> None:
    """Instantiate the right pipeline (online / offline) and drain events."""
    config = _build_pipeline_config(args)
    requested_backend = getattr(args, "diar_backend", "auto")
    # Only the ``file`` subcommand carries the batch/offline levers ; mic/url
    # are inherently live and always take the streaming path.
    if hasattr(args, "no_real_time") or getattr(args, "offline", False):
        # A file: the length is cheap to read, so the router can fire its
        # short→nemo / long→pyannote crossover instead of guessing.
        use_offline, diar_cfg, note = _choose_file_diar(
            config.diar,
            explicit_offline=getattr(args, "offline", False),
            batch=getattr(args, "no_real_time", False),
            force_online=getattr(args, "online", False),
            duration_s=getattr(args, "duration_s", None),
            requested_backend=requested_backend,
        )
        if note:
            sys.stderr.write(note + "\n")
    else:
        # Live mic / URL: no duration, inherently streaming — still route the
        # backend (auto → nemo per the study) instead of leaking ``"auto"``.
        use_offline = False
        backend, note = _route_backend(
            requested_backend=requested_backend, live=True, duration_s=None
        )
        if note:
            sys.stderr.write(note + "\n")
        diar_cfg = {**config.diar, "backend": backend}

    pipeline: Pipeline | OfflinePipeline
    if use_offline:
        # Offline pipeline skips the VAD and gives the diar backend the whole buffer.
        # We re-project the shared config onto the offline shape (no ``vad`` /
        # ``eot`` block) so the two pipelines stay driven by one CLI namespace.
        offline_cfg = OfflinePipelineConfig(
            diar=diar_cfg,
            asr=config.asr,
            llm=config.llm,
        )
        pipeline = OfflinePipeline(source=source_factory, config=offline_cfg)
    else:
        config.diar = diar_cfg
        pipeline = Pipeline(source=source_factory, config=config)
    # Drain the async event stream synchronously — one line per event as it lands.
    async for ev in pipeline.run():
        _print_event(ev, args.jsonl)


def _handle_mic(args: argparse.Namespace) -> int:
    """Handle ``vocal-helper mic`` — stream the live microphone through the pipeline."""
    # Import lazily so users without the [mic] extra can still use file/url.
    from vocal_helper.sources import from_microphone

    def factory() -> AsyncIterator[PcmFrame]:
        """Open a fresh 16 kHz / 20 ms microphone stream on each pipeline start."""
        # 16 kHz mono matches whisper.cpp's native rate ; 20 ms frames are the
        # Silero VAD stride, so no downstream resampling / reframing is needed.
        return from_microphone(
            device_name=args.device,
            sample_rate=16_000,
            frame_ms=20,
        )

    # The whole pipeline lives inside one event loop for this process.
    asyncio.run(_run_pipeline(args, factory))
    return 0


def _handle_file(args: argparse.Namespace) -> int:
    """Handle ``vocal-helper file`` — replay a WAV file through the pipeline."""
    from vocal_helper.sources import from_wav_file, probe_duration_s

    path = Path(args.path)
    # Probe the length once (cheap ffprobe metadata read) so the diarization
    # router can fire its short→nemo / long→pyannote crossover on the real
    # duration instead of the decorative ``duration_s=None`` fallback.
    args.duration_s = probe_duration_s(path)

    def factory() -> AsyncIterator[PcmFrame]:
        """Open the WAV source ; honour ``--no-real-time`` to skip wall-clock pacing."""
        # Real-time pacing mimics a live feed (useful for demos / latency numbers) ;
        # ``--no-real-time`` fires frames as fast as they decode for batch throughput.
        return from_wav_file(path, real_time=not args.no_real_time)

    asyncio.run(_run_pipeline(args, factory))
    return 0


def _handle_url(args: argparse.Namespace) -> int:
    """Handle ``vocal-helper url`` — stream any yt-dlp-reachable URL through the pipeline."""
    # URL streaming needs podcast_helper (the [stream] extra).
    from vocal_helper.sources import from_url

    def factory() -> AsyncIterator[PcmFrame]:
        """Open a streaming source for the given URL (yt-dlp resolves the media)."""
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
        choices=["auto", "pyannote", "nemo", "sherpa"],
        default="auto",
        help="Speaker-diarization backend. Default 'auto' delegates to the "
        "study-grounded router (the aiguilleur): offline short (≤300 s, ≤4 "
        "speakers) → 'nemo', offline long/unknown → 'pyannote', live stream → "
        "'nemo', reporting both DER (quality) and RTF (speed). Pass an explicit "
        "'pyannote' / 'nemo' / 'sherpa' to override; 'sherpa' is torch-free "
        "(`pip install vocal-helper[sherpa]`).",
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
        help="Batch mode: process as fast as possible (skip real-time pacing). "
        "By default this auto-selects the offline pyannote diarizer when its "
        "bundle is present — the most reliable path (DER ~0.12 on AMI vs ~0.50 "
        "for the online diarizer, 2026-07-16 sweep). If the bundle is absent it "
        "falls back to the online diarizer with the global re-clustering repair "
        "pass. Pass --online to force the streaming diarizer.",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Force the OfflinePipeline (pyannote 3.1 whole-buffer, global "
        "clustering) — the most reliable path on meetings, podcasts, lectures. "
        "Honours --diar-backend. This is already the default for --no-real-time "
        "when the bundle is available.",
    )
    p.add_argument(
        "--online",
        action="store_true",
        help="Force the online streaming diarizer for a batch file run instead "
        "of auto-selecting offline (lighter, lower latency, but higher DER).",
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
    # Every subparser wired a ``func`` handler via ``set_defaults`` ; dispatch to
    # it and normalise the return to a plain int exit code for ``SystemExit``.
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
