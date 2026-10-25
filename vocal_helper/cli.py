"""
Vocal Helper — the ``vocal-helper`` command-line interface.

Thin wrapper around the async :class:`vocal_helper.Pipeline` /
:class:`vocal_helper.OfflinePipeline` orchestrators that exposes the
toolkit as subcommands under a single entry point, built with
:mod:`click`.

Subcommands
-----------
- ``mic``        — live microphone input (needs the ``[mic]`` extra)
- ``file``       — replay a mono 16 kHz WAV through the pipeline
- ``url``        — stream from any URL yt-dlp can reach (``[stream]`` extra)
- ``transcribe`` — one-shot transcription of a WAV file

Every pipeline subcommand accepts a common core of ``--whisper-model`` /
``--language`` / ``--diar-backend`` / ``--llm`` / ``--jsonl`` levers ; see
``vocal-helper <subcommand> --help`` for the full surface.

Design notes
------------
- Async work is wrapped in :func:`asyncio.run` inside each command ;
  click itself stays sync.
- This module also owns the diarization decisions the HTTP surface shares
  (:func:`_route_backend`, :func:`_choose_file_diar`), so the
  study-grounded router is enforced from exactly one place.

Usage Example
-------------
>>> #   vocal-helper mic --llm --jsonl
>>> #   vocal-helper file meeting.wav --offline --language en
>>> #   vocal-helper url "https://youtu.be/…" --language fr
>>> #   vocal-helper transcribe clip.wav --language en

Author
------
Warith Harchaoui, Ph.D. , https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

import click

from vocal_helper import __version__
from vocal_helper.pipeline import (
    OfflinePipeline,
    OfflinePipelineConfig,
    Pipeline,
    PipelineConfig,
    resolve_engine,
)
from vocal_helper.router import select_diarization
from vocal_helper.types import PcmFrame

# ---------------------------------------------------------------------------
# Config translation — the one place mapping CLI options onto a
# PipelineConfig. We keep the click callbacks small by punching the shared
# kwargs through a plain dict.
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
    llm_recent_window_s: float,
    endpoint: str | None,
    eot: bool,
) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from the shared click options."""
    # ASR dict passed straight through to WhisperStage.__init__ — the keys are
    # that constructor's parameter names, not a CLI-local shape.
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
    # Only override the join threshold when the user passed one — otherwise let
    # the diarizer keep its tuned 0.30 default rather than pinning it here.
    if join_threshold is not None:
        diar_cfg["join_threshold"] = join_threshold
    # LLM analyst and semantic-EOT stages are opt-in and both need an LLM. The
    # model is never a flag — best-engine-ai-helper resolves it (per machine)
    # from the committed ``llm.brief.yaml``. Resolve once, thread the same
    # engine descriptor into whichever stage(s) are enabled.
    llm_cfg: dict | None = None
    eot_cfg: dict | None = None
    if llm or eot:
        engine = resolve_engine(endpoint=endpoint)
        if llm:
            llm_cfg = {"engine": engine, "recent_window_s": llm_recent_window_s}
        # Keys must match SemanticEOTStage.__init__ (``engine``).
        if eot:
            eot_cfg = {"engine": engine}
    return PipelineConfig(diar=diar_cfg, asr=asr_cfg, llm=llm_cfg, eot=eot_cfg)


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

    The one place the study-grounded router (the *aiguilleur*) is enforced for
    files, shared by the ``file`` command and the HTTP surface so the decision
    cannot drift between them. The offline whole-buffer path is the reliability
    default for batch (2026-07-16 DER sweep: offline 0.12 / 0.34 vs online
    0.50 / 0.59); *which* offline backend runs is the router's call, fed the
    file's real probed ``duration_s`` so the short→nemo / long→pyannote
    crossover actually fires. Precedence:

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


def _resolve_online_backend(cfg: PipelineConfig, requested_backend: str) -> None:
    """Resolve a live command's ``"auto"`` diar backend through the router, in place.

    The ``mic`` / ``url`` commands build a streaming :class:`Pipeline` directly,
    so — unlike ``file`` — they never pass through :func:`_choose_file_diar`. This
    routes their backend the same way (live stream, no duration → ``nemo`` per the
    study, or an explicit override) and mutates ``cfg.diar['backend']`` so the
    sentinel ``"auto"`` never reaches :class:`~vocal_helper.OnlineDiarStage`.

    Parameters
    ----------
    cfg : PipelineConfig
        The live pipeline config whose ``diar`` backend is resolved in place.
    requested_backend : str
        ``"auto"`` to route, or an explicit backend name to honour as override.

    Returns
    -------
    None
        ``cfg.diar['backend']`` is updated as a side effect; any router note is
        written to stderr.
    """
    backend, note = _route_backend(requested_backend=requested_backend, live=True, duration_s=None)
    if note:
        sys.stderr.write(note + "\n")
    cfg.diar["backend"] = backend


def _print_event(ev: Mapping[str, Any], jsonl: bool) -> None:
    """Emit a single pipeline event to stdout, JSONL or human-readable."""
    if jsonl:
        # Strip the raw PCM before serialising — cheaper log, right transport.
        sys.stdout.write(json.dumps({k: v for k, v in ev.items() if k != "pcm"}) + "\n")
        sys.stdout.flush()
        return
    # Human-readable branch — the event's shape tells us which template to use:
    # a ``text`` key is an utterance line, a ``summary`` key is a rolling digest.
    if "text" in ev:
        sys.stdout.write(f"[{ev['t0']:7.2f}s -> {ev['t1']:7.2f}s  {ev['speaker']}]  {ev['text']}\n")
    elif "summary" in ev:
        sys.stdout.write(
            f"\n--- rolling summary @ {ev['t0']:.1f}s "
            f"(model={ev['model']}) ---\n{ev['summary']}\n"
            f"--- recent window ---\n{ev['recent']}\n\n"
        )
    sys.stdout.flush()


async def _drain(pipeline: Pipeline | OfflinePipeline, jsonl: bool) -> None:
    """Consume every event emitted by ``pipeline.run()``.

    Parameters
    ----------
    pipeline : Pipeline or OfflinePipeline
        A started orchestrator whose ``run()`` coroutine yields event dicts.
    jsonl : bool
        When ``True`` each event is printed as a single JSON line ; otherwise
        it is rendered in the human-readable format.

    Returns
    -------
    None
        Events are written to stdout as a side effect ; nothing is returned.
    """
    # Pull events as they land and hand each straight to the printer — the
    # pipeline back-pressures us, so this loop paces the whole CLI.
    async for ev in pipeline.run():
        _print_event(ev, jsonl)


# ---------------------------------------------------------------------------
# Reusable option bundle — Click does not have a built-in shared-options
# decorator, but we can compose one with a helper. Every subcommand slaps
# ``@_common_options`` on top of its own signature.
# ---------------------------------------------------------------------------


def _common_options(func: Callable[..., object]) -> Callable[..., object]:
    """Decorator adding the shared VAD / diar / ASR / LLM levers.

    Parameters
    ----------
    func : Callable[..., object]
        The click command callback to wrap with the shared option stack.

    Returns
    -------
    Callable[..., object]
        The same callback with every shared ``click.option`` applied, so it
        can be composed on top of a subcommand's own signature.
    """
    # Order matters for --help output; the stack below is the order users read.
    func = click.option(
        "--jsonl", is_flag=True, default=False, help="Emit one JSON event per line."
    )(func)
    func = click.option(
        "--eot",
        is_flag=True,
        default=False,
        help="Enable the SemanticEOTStage (LiveKit-style turn "
        "detector). Reduces mid-sentence cuts at the cost of "
        "one extra LLM hop per voiced segment.",
    )(func)
    func = click.option(
        "--endpoint",
        default=None,
        help="Override the LLM serving base URL passed to best-engine-ai-helper "
        "(e.g. a remote Ollama / vLLM host).",
    )(func)
    func = click.option(
        "--llm-recent-window-s",
        type=float,
        default=60.0,
        show_default=True,
        help="Verbatim window (seconds) kept out of the summary.",
    )(func)
    func = click.option("--llm", is_flag=True, default=False, help="Enable the LLM analyst stage.")(
        func
    )
    func = click.option(
        "--join-threshold",
        type=float,
        default=None,
        help="Cosine-distance join threshold for online diarizer (default 0.30).",
    )(func)
    func = click.option(
        "--diar-backend",
        type=click.Choice(["auto", "pyannote", "nemo", "sherpa"]),
        default="auto",
        show_default=True,
        help="Speaker-diarization backend. Default 'auto' delegates to the "
        "study-grounded router (the aiguilleur): offline short → 'nemo', "
        "offline long/unknown → 'pyannote', live → 'nemo', reporting DER "
        "(quality) and RTF (speed). Pass 'pyannote' / 'nemo' / 'sherpa' to "
        "override; 'sherpa' is torch-free (needs the [sherpa] extra).",
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
# ``__version__`` is passed explicitly rather than letting click read the
# installed distribution metadata: in an editable checkout that metadata
# reports whatever was last pip-installed, so ``-V`` would lie about the
# source you are actually running.
@click.version_option(__version__, "-V", "--version", prog_name="vocal-helper")
def cli() -> None:
    """Turn audio into diarized, transcribed utterances — live or from a file.

    Each subcommand feeds one source (microphone, WAV, URL) through the
    VAD → diarization → ASR pipeline, with an optional rolling LLM summary.
    The "transcribe" command is the one-shot escape hatch: no VAD, no diarization.
    """
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
    llm_recent_window_s: float,
    endpoint: str | None,
    eot: bool,
    jsonl: bool,
    device: str | None,
) -> None:
    """Live microphone input (needs the [mic] extra)."""
    from vocal_helper.sources import from_microphone

    cfg = _pipeline_config(
        whisper_model=whisper_model,
        language=language,
        threads=threads,
        initial_prompt=initial_prompt,
        diar_backend=diar_backend,
        join_threshold=join_threshold,
        llm=llm,
        llm_recent_window_s=llm_recent_window_s,
        endpoint=endpoint,
        eot=eot,
    )

    def factory() -> AsyncIterator[PcmFrame]:
        """Open a fresh 16 kHz / 20 ms microphone stream on each pipeline start."""
        # 16 kHz mono is whisper.cpp's native rate ; 20 ms is the Silero VAD stride,
        # so the source hands the pipeline frames it can consume without resampling.
        return from_microphone(device_name=device, sample_rate=16_000, frame_ms=20)

    # A live stream carries no duration — route the online backend (auto → nemo
    # per the study) so ``"auto"`` never reaches the stage.
    _resolve_online_backend(cfg, diar_backend)
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
    help="Batch mode: process as fast as possible (skip real-time pacing). By "
    "default auto-selects the offline pyannote diarizer when its bundle is "
    "present — the most reliable path (DER ~0.12 on AMI vs ~0.50 online, "
    "2026-07-16 sweep) — else falls back to the online diarizer with the global "
    "re-clustering repair pass. Pass --online to force the streaming diarizer.",
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help="Force the OfflinePipeline (pyannote 3.1 whole-buffer, global "
    "clustering) — most reliable on meetings/podcasts/lectures. Honours "
    "--diar-backend. Already the default for --no-real-time when the bundle is "
    "available.",
)
@click.option(
    "--online",
    is_flag=True,
    default=False,
    help="Force the online streaming diarizer for a batch file run instead of "
    "auto-selecting offline (lighter, lower latency, higher DER).",
)
def file(
    whisper_model: str,
    language: str,
    threads: int,
    initial_prompt: str,
    diar_backend: str,
    join_threshold: float | None,
    llm: bool,
    llm_recent_window_s: float,
    endpoint: str | None,
    eot: bool,
    jsonl: bool,
    path: str,
    no_real_time: bool,
    offline: bool,
    online: bool,
) -> None:
    """Replay a 16 kHz mono WAV through the pipeline."""
    from vocal_helper.sources import from_wav_file, probe_duration_s

    cfg = _pipeline_config(
        whisper_model=whisper_model,
        language=language,
        threads=threads,
        initial_prompt=initial_prompt,
        diar_backend=diar_backend,
        join_threshold=join_threshold,
        llm=llm,
        llm_recent_window_s=llm_recent_window_s,
        endpoint=endpoint,
        eot=eot,
    )

    def factory() -> AsyncIterator[PcmFrame]:
        """Open the WAV source ; ``--no-real-time`` skips wall-clock pacing for batch runs."""
        # Real-time pacing simulates a live feed ; disabling it fires frames as fast
        # as they decode, which is what you want when timing throughput on a file.
        return from_wav_file(Path(path), real_time=not no_real_time)

    # Enforce the router (see _choose_file_diar above):
    # batch runs prefer the offline whole-buffer diarizer when a backend is
    # installed, with the study-grounded router choosing that backend from the
    # file's real probed duration — short→nemo, long→pyannote — reporting DER +
    # RTF. Falls back to the online diarizer + refine pass when no offline
    # backend is available.
    use_offline, diar_cfg, note = _choose_file_diar(
        cfg.diar,
        explicit_offline=offline,
        batch=no_real_time,
        force_online=online,
        duration_s=probe_duration_s(Path(path)),
        requested_backend=diar_backend,
    )
    if note:
        sys.stderr.write(note + "\n")
    pipeline: Pipeline | OfflinePipeline
    if use_offline:
        pipeline = OfflinePipeline(
            source=factory,
            config=OfflinePipelineConfig(diar=diar_cfg, asr=cfg.asr, llm=cfg.llm),
        )
    else:
        cfg.diar = diar_cfg
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
    llm_recent_window_s: float,
    endpoint: str | None,
    eot: bool,
    jsonl: bool,
    url: str,
) -> None:
    """Stream from any URL yt-dlp can reach (needs the [stream] extra)."""
    from vocal_helper.sources import from_url as _from_url

    cfg = _pipeline_config(
        whisper_model=whisper_model,
        language=language,
        threads=threads,
        initial_prompt=initial_prompt,
        diar_backend=diar_backend,
        join_threshold=join_threshold,
        llm=llm,
        llm_recent_window_s=llm_recent_window_s,
        endpoint=endpoint,
        eot=eot,
    )

    def factory() -> AsyncIterator[PcmFrame]:
        """Open a streaming source for the given URL (yt-dlp resolves the media)."""
        return _from_url(url)

    # URL playback is a live stream — route the online backend (auto → nemo).
    _resolve_online_backend(cfg, diar_backend)
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


def main() -> None:
    """Console entry point (``vocal-helper``).

    click's own ``main()`` only special-cases ``ClickException``/``Abort``
    (and a broken pipe); a plain library exception would otherwise
    propagate as a raw Python traceback instead of a clean CLI error. This
    wraps the whole invocation and translates that last case into a
    one-line stderr message + exit 1 — click's own control flow (usage
    errors, ``--help``, an explicit ``sys.exit`` in a subcommand) already
    raises ``SystemExit``, a ``BaseException`` this does not catch, so it
    passes through untouched.
    """
    try:
        cli()
    except Exception as err:  # noqa: BLE001 — last resort: see docstring
        click.echo(f"Error: {err}", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
