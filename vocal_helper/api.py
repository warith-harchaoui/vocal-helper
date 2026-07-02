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

try:
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The FastAPI HTTP surface requires the [api] extra. "
        "Install with: pip install 'vocal-helper[api]'"
    ) from exc


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
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


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
    ext = suffix_hint or (Path(upload.filename or "").suffix or ".wav")
    if not ext.startswith("."):
        ext = "." + ext
    out = dest_dir / (f"upload{ext}")
    with out.open("wb") as fp:
        shutil.copyfileobj(upload.file, fp)
    return out


def _cleanup(*paths: Path | str) -> None:
    """Best-effort temp cleanup — never let a tidy-up failure kill a response."""
    for p in paths:
        try:
            path = Path(p)
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)
        except Exception:
            pass


def _new_tmpdir() -> Path:
    """Create a request-scoped temp directory under the system temp root."""
    return Path(tempfile.mkdtemp(prefix="vocal-helper-"))


def _load_pcm_mono_16k(path: Path):
    """
    Load an audio file as a mono float32 numpy array at 16 kHz.

    We route through ``audio_helper.sound_converter`` when the source is
    not already a 16 kHz mono WAV, so all upstream codecs (mp3 / m4a /
    ogg / flac / opus / …) work.
    """
    import numpy as np
    import soundfile as sf

    # Soundfile handles WAV / FLAC / OGG natively; for other formats we
    # transcode with audio_helper (ffmpeg-backed) into a temp WAV.
    try:
        pcm, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        try:
            from audio_helper import sound_converter
        except ImportError as exc:  # pragma: no cover
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {path.suffix}. "
                       f"Install audio-helper (already a dependency) or upload WAV/FLAC/OGG.",
            ) from exc
        wav = path.with_suffix(".vh.wav")
        sound_converter(
            input_audio=str(path), output_audio=str(wav),
            freq=16_000, channels=1, encoding="pcm_s16le", overwrite=True,
        )
        pcm, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    # Down-mix any stereo left behind by an odd container to mono.
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1).astype(np.float32)
    if sr != 16_000:
        # Cheap linear resample — good enough for whisper's own front-end.
        import numpy as np

        n_out = int(round(pcm.shape[0] * 16_000 / sr))
        pcm = np.interp(
            np.linspace(0.0, 1.0, n_out, endpoint=False),
            np.linspace(0.0, 1.0, pcm.shape[0], endpoint=False),
            pcm,
        ).astype(np.float32)
        sr = 16_000
    return pcm, sr


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
    from vocal_helper.asr import transcribe_pcm

    tmp = _new_tmpdir()
    try:
        src = _spool(file, tmp)
        pcm, sr = _load_pcm_mono_16k(src)
        text = transcribe_pcm(
            pcm=pcm, sr=int(sr),
            model=whisper_model, language=language, threads=threads,
        )
    finally:
        # Text response — synchronous cleanup, no background handler needed.
        _cleanup(tmp)
    return JSONResponse({"text": text, "language": language})


@app.post("/pipeline", tags=["actions"])
def pipeline(
    background: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (WAV / MP3 / M4A / OGG / FLAC)."),
    language: str = Form("auto"),
    whisper_model: str = Form("large-v3-turbo-q5_0"),
    threads: int = Form(6),
    diar_backend: str = Form("pyannote", description="pyannote | nemo"),
    hf_token: str | None = Form(None, description="HuggingFace token for pyannote."),
    llm: bool = Form(False, description="Enable the Gemma analyst stage."),
    llm_model: str = Form("gemma4:e4b"),
    llm_recent_window_s: float = Form(60.0),
) -> JSONResponse:
    """Run the full OfflinePipeline on the uploaded file and return the events."""
    # Local imports so ``pip install vocal-helper[api]`` alone (without the
    # pyannote / nemo extras) still boots the server for /health probes.
    import numpy as np

    from vocal_helper._settings import resolve_hf_token
    from vocal_helper.pipeline import OfflinePipeline, OfflinePipelineConfig
    from vocal_helper.sources import from_numpy_array

    tmp = _new_tmpdir()
    try:
        src = _spool(file, tmp)
        pcm, sr = _load_pcm_mono_16k(src)
        asr_cfg: dict = {"model": whisper_model, "language": language, "threads": threads}
        diar_cfg: dict = {"backend": diar_backend}
        token = resolve_hf_token(hf_token)
        if token:
            diar_cfg["hf_token"] = token
        llm_cfg: dict | None = None
        if llm:
            llm_cfg = {"model": llm_model, "recent_window_s": llm_recent_window_s}
        cfg = OfflinePipelineConfig(diar=diar_cfg, asr=asr_cfg, llm=llm_cfg)

        def factory():
            return from_numpy_array(
                np.asarray(pcm, dtype=np.float32),
                sample_rate=int(sr),
                frame_ms=20,
            )

        pipe = OfflinePipeline(source=factory, config=cfg)

        async def collect() -> list[dict]:
            events: list[dict] = []
            async for ev in pipe.run():
                # Strip raw PCM (float32 arrays don't JSON-serialise) — the
                # timestamps + speaker + text are what the caller wanted.
                events.append({k: v for k, v in ev.items() if k != "pcm"})
            return events

        events = asyncio.run(collect())
    finally:
        _cleanup(tmp)
    return JSONResponse({"events": events, "count": len(events)})
