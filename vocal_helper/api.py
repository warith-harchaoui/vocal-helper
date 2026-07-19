"""
Vocal Helper — FastAPI HTTP surface.

Exposes the offline batch pipeline over HTTP so ``vocal-helper`` can be
dropped behind any reverse proxy and consumed by other services. The
online streaming pipeline lives on-process (queues + coroutines) and is
not exposed as REST endpoints — it is either driven from the CLI, the
Python API, or a WebSocket surface that lives elsewhere.

What ships here
---------------
- ``POST /transcribe`` — one-shot ASR of an uploaded WAV / mp3 / m4a /
  ogg / flac. Returns ``{"text": ..., "language": ...}``.
- ``POST /pipeline`` — full offline pipeline (VAD + diarization + STT
  + optional Gemma summary) on an uploaded audio file. Returns a list
  of :class:`Utterance` events, optionally with the final summary.
- ``GET /health`` — liveness probe.

Install the extra to get the runtime dependencies::

    pip install 'vocal-helper[api]'

Then run the app with any ASGI server::

    uvicorn vocal_helper.api:app --host 0.0.0.0 --port 8000

Usage Example
-------------
>>> # Start the server:
>>> #   uvicorn vocal_helper.api:app --reload
>>> # One-shot transcription:
>>> #   curl -F 'file=@clip.wav' -F 'language=en' \\
>>> #        http://localhost:8000/transcribe
>>> # Full offline pipeline:
>>> #   curl -F 'file=@meeting.wav' -F 'llm=true' \\
>>> #        http://localhost:8000/pipeline
>>> # Full OpenAPI docs at http://localhost:8000/docs

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

# ``numpy`` is imported lazily inside the request path (it is a heavy import
# and the meta / health endpoints never touch it). Under ``TYPE_CHECKING``
# we still pull the array types so annotations resolve for type-checkers
# without paying the import cost at runtime.
if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import numpy as np
    from numpy.typing import NDArray

    from vocal_helper.types import PcmFrame

try:
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The FastAPI HTTP surface requires the [api] extra. "
        "Install with: pip install 'vocal-helper[api]'"
    ) from exc

# Single source of truth for the version — read it from the package so the
# HTTP surface can never drift from ``pyproject`` / ``__init__`` again (it was
# once stale at 0.3.7 while the package was several releases ahead).
from vocal_helper import __version__ as _VERSION

# ---------------------------------------------------------------------------
# App factory + shared plumbing
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Vocal Helper API",
    description=(
        "HTTP surface for the vocal-helper offline pipeline: one-shot ASR "
        "and full VAD + diarization + STT + optional Gemma summary on an "
        "uploaded audio file."
    ),
    version=_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Optionally serve the minimal local web GUI (repo ``webui/`` folder) at ``/ui``
# when it is present next to the package (a source checkout). Mounting it on the
# API itself keeps it **same-origin** — the page's ``fetch`` hits ``/transcribe``
# with no CORS dance — and fully local. A pip install without the folder simply
# skips the mount, so this never breaks a headless server.
_WEBUI_DIR = Path(__file__).resolve().parent.parent / "webui"
if _WEBUI_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/ui", StaticFiles(directory=str(_WEBUI_DIR), html=True), name="webui")


def _spool(upload: UploadFile, dest_dir: Path, suffix_hint: str | None = None) -> Path:
    """
    Persist an ``UploadFile`` to a temp path on disk.

    We copy the stream rather than holding bytes in memory — meeting-length
    audio is routinely > 100 MB and we want the worker to survive.

    Parameters
    ----------
    upload : UploadFile
        The FastAPI upload object.
    dest_dir : Path
        Temp directory that will hold the spooled file.
    suffix_hint : str, optional
        Extension override. Falls back to the client-provided filename's
        suffix, then to ``.wav``.

    Returns
    -------
    Path
        Path to the spooled file on disk.
    """
    # Prefer the explicit hint, else the client filename's suffix, else WAV.
    # The suffix matters : ffmpeg/audio-helper sniff container by extension.
    ext = suffix_hint or (Path(upload.filename or "").suffix or ".wav")
    # Normalise a bare extension ("wav" → ".wav") so the join below is safe.
    if not ext.startswith("."):
        ext = "." + ext
    out = dest_dir / (f"upload{ext}")
    # ``copyfileobj`` streams in fixed-size chunks — constant memory even for
    # a multi-hundred-MB meeting recording, unlike ``read()`` + ``write()``.
    with out.open("wb") as fp:
        shutil.copyfileobj(upload.file, fp)
    return out


def _cleanup(*paths: Path | str) -> None:
    """Best-effort temp cleanup — never let a tidy-up failure kill a response."""
    for p in paths:
        # Swallow every error : cleanup runs in a ``finally`` / background
        # task, so a failed unlink must never mask the real response or 500.
        try:
            path = Path(p)
            # Directory vs file need different removal calls ; branch on it.
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)
        except Exception:
            pass


def _new_tmpdir() -> Path:
    """Create a request-scoped temp directory under the system temp root."""
    return Path(tempfile.mkdtemp(prefix="vocal-helper-"))


def _load_pcm_mono_16k(path: Path) -> tuple[NDArray[np.float32], int]:
    """
    Load an audio file as a mono float32 numpy array at 16 kHz.

    Decoding goes through ``audio_helper.load_audio`` (ffmpeg-backed), so
    every upstream codec / container works — mp3, m4a/AAC, ogg, flac, opus,
    and the audio track of video files — never libsndfile / soundfile.

    Parameters
    ----------
    path : Path
        Filesystem path to the uploaded audio (any ffmpeg-decodable codec /
        container).

    Returns
    -------
    tuple[NDArray[np.float32], int]
        The mono float32 waveform of shape ``(n_samples,)`` and its sample
        rate (always ``16000``).

    Raises
    ------
    HTTPException
        With status 400 when ``audio-helper`` is missing or the file cannot
        be decoded (treated as a bad client upload).
    """
    import numpy as np

    try:
        from audio_helper import load_audio
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decode {path.suffix}: audio-helper is required.",
        ) from exc
    # 16 kHz mono float32 is the contract every downstream stage (VAD /
    # diarization / whisper) is trained on — enforce it here at ingest.
    try:
        pcm, sr = load_audio(str(path), target_sample_rate=16_000, to_mono=True, to_numpy=True)
    except Exception as exc:
        # A decode failure is a client problem (bad / unsupported upload),
        # so surface it as 400 rather than letting it bubble up as a 500.
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio {path.name!r}: {exc}",
        ) from exc
    return np.asarray(pcm, dtype=np.float32), int(sr)


def _resolve_offline_backend(diar_backend: str, n_samples: int, sr: int) -> str:
    """Resolve the offline diarization backend for an uploaded file via the router.

    The HTTP ``/pipeline`` (and, through it, the MCP tool) choke-point that makes
    the *aiguilleur* actually enforced server-side: ``"auto"`` hands the decoded
    buffer's real duration to :func:`~vocal_helper.router.select_diarization`
    (offline, ``live=False``) so short/dense audio routes to ``nemo`` and long
    audio to ``pyannote``, subject to which backends are installed. Any explicit
    backend is honoured verbatim.

    Parameters
    ----------
    diar_backend : str
        ``"auto"`` to route, or ``"pyannote"`` / ``"nemo"`` / ``"sherpa"``.
    n_samples : int
        Number of PCM samples in the decoded mono buffer.
    sr : int
        Sample rate in Hz (``0`` ⇒ unknown, treated as unknown duration).

    Returns
    -------
    str
        The concrete backend name to hand the offline diarizer.

    Examples
    --------
    >>> _resolve_offline_backend("pyannote", 16_000, 16_000)
    'pyannote'
    """
    # An explicit backend is the caller's override — never second-guess it.
    if diar_backend != "auto":
        return diar_backend
    # Import lazily so the base [api] install (no pyannote/nemo extras) still
    # boots for /health without pulling the availability probes' dependencies.
    from vocal_helper.cli_argparse import _offline_nemo_available, _offline_pyannote_available
    from vocal_helper.router import select_diarization

    # Duration from the decoded buffer — O(1), already in memory. sr==0 (unknown)
    # collapses to None so the router takes its safe long-form branch.
    duration_s = float(n_samples) / float(sr) if sr else None
    plan = select_diarization(
        live=False,
        duration_s=duration_s,
        pyannote_available=_offline_pyannote_available(),
        nemo_available=_offline_nemo_available(),
    )
    return plan.backend


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Simple liveness probe — no dependency check, just proves the app is up."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@app.post("/transcribe", tags=["actions"])
def transcribe(
    background: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (WAV / MP3 / M4A / OGG / FLAC)."),
    language: str = Form("auto", description="ISO-639-1 code or 'auto' for language ID."),
    whisper_model: str = Form("large-v3-turbo-q5_0"),
    threads: int = Form(6),
) -> JSONResponse:
    """One-shot transcription of the uploaded audio — no VAD, no diarization."""
    # Import inside the handler so a bare [api] install still serves /health
    # even when the heavier ASR backend isn't wired up yet.
    from vocal_helper.asr import transcribe_pcm_with_language

    # Spool → decode → transcribe, all rooted in one request-scoped tmpdir
    # so the ``finally`` cleanup below can remove everything in one shot.
    tmp = _new_tmpdir()
    try:
        src = _spool(file, tmp)
        pcm, sr = _load_pcm_mono_16k(src)
        # Capture the language whisper actually used, not the request input:
        # with ``language="auto"`` that is the language discovered from the
        # audio, so the response tells the truth instead of echoing "auto".
        text, detected = transcribe_pcm_with_language(
            pcm=pcm,
            sr=int(sr),
            model=whisper_model,
            language=language,
            threads=threads,
        )
    finally:
        # Text response — synchronous cleanup, no background handler needed.
        _cleanup(tmp)
    # Fall back to the request value only when whisper reported no language
    # (e.g. an empty upload) so the field is never silently null on success.
    return JSONResponse({"text": text, "language": detected or language})


@app.post("/pipeline", tags=["actions"])
def pipeline(
    background: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (WAV / MP3 / M4A / OGG / FLAC)."),
    language: str = Form("auto"),
    whisper_model: str = Form("large-v3-turbo-q5_0"),
    threads: int = Form(6),
    diar_backend: str = Form(
        "auto",
        description="auto | pyannote | nemo | sherpa. 'auto' lets the router pick "
        "by duration (short→nemo, long→pyannote), reporting DER + RTF.",
    ),
    llm: bool = Form(False, description="Enable the Gemma analyst stage."),
    llm_model: str = Form("gemma4:e4b"),
    llm_recent_window_s: float = Form(60.0),
) -> JSONResponse:
    """Run the full OfflinePipeline on the uploaded file and return the events."""
    # Local imports so ``pip install vocal-helper[api]`` alone (without the
    # pyannote / nemo extras) still boots the server for /health probes.
    import numpy as np

    from vocal_helper.pipeline import OfflinePipeline, OfflinePipelineConfig
    from vocal_helper.sources import from_numpy_array

    tmp = _new_tmpdir()
    try:
        src = _spool(file, tmp)
        pcm, sr = _load_pcm_mono_16k(src)
        asr_cfg: dict = {"model": whisper_model, "language": language, "threads": threads}
        # Enforce the study-grounded router (the aiguilleur): the decoded buffer's
        # real duration drives the offline backend choice — short/dense → nemo,
        # long → pyannote — unless the caller pins an explicit backend.
        resolved_backend = _resolve_offline_backend(diar_backend, len(pcm), int(sr))
        # Model weights load from the self-hosted diarization-engines bundle
        # (settings.yaml ``engines.diarization_url``) — no HuggingFace token.
        diar_cfg: dict = {"backend": resolved_backend}
        # The LLM analyst stage is opt-in : only attach its config when the
        # caller asked for it, otherwise leave it ``None`` (stage skipped).
        llm_cfg: dict | None = None
        if llm:
            llm_cfg = {"model": llm_model, "recent_window_s": llm_recent_window_s}
        cfg = OfflinePipelineConfig(diar=diar_cfg, asr=asr_cfg, llm=llm_cfg)

        def factory() -> AsyncIterator[PcmFrame]:
            """Build a fresh PcmFrame source over the decoded buffer.

            The pipeline takes a *callable* (not a live iterator) so it can
            re-create the source per run ; we close over the decoded PCM.
            """
            # ``frame_ms=20`` matches the pipeline's expected framing.
            return from_numpy_array(
                np.asarray(pcm, dtype=np.float32),
                sample_rate=int(sr),
                frame_ms=20,
            )

        pipe = OfflinePipeline(source=factory, config=cfg)

        async def collect() -> list[dict]:
            """Drain the pipeline's async event stream into a JSON-ready list."""
            events: list[dict] = []
            async for ev in pipe.run():
                # Strip raw PCM (float32 arrays don't JSON-serialise) — the
                # timestamps + speaker + text are what the caller wanted.
                events.append({k: v for k, v in ev.items() if k != "pcm"})
            return events

        # The pipeline is async but this handler is sync — bridge with a
        # one-shot event loop that runs to completion before we respond.
        events = asyncio.run(collect())
    finally:
        _cleanup(tmp)
    return JSONResponse({"events": events, "count": len(events)})
