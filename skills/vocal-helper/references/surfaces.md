# vocal-helper non-CLI surfaces

The same pipeline is reachable through five surfaces. The Python library and
argparse CLI are always available; the others live behind optional extras.

## 1. Python library (default)

```python
import vocal_helper as voh

# Online streaming pipeline (mic / URL) — producer/consumer coroutines.
voh.Pipeline(source=..., config=voh.PipelineConfig(...))
# Offline batch pipeline (a whole decoded file).
voh.OfflinePipeline(source=..., config=voh.OfflinePipelineConfig(...))

# One-shot ASR helpers.
voh.transcribe_pcm(pcm, sr, model=..., language="auto", threads=6)         # -> str
voh.transcribe_pcm_with_language(pcm, sr, ...)                             # -> (text, lang)

# Stages (compose your own graph).
voh.SileroVADStage(...)          # voice activity detection
voh.OnlineDiarStage(...) / voh.OfflineDiarStage(...)   # who spoke when
voh.WhisperStage(...)            # STT
voh.GemmaAnalystStage(...)       # rolling local-LLM summary

# The study-grounded backend router.
voh.select_diarization(live=False, duration_s=..., pyannote_available=..., nemo_available=...)  # -> BackendPlan

# Spoken-language identification.
voh.detect_language(pcm, sr)                # -> str
voh.detect_language_regions(pcm, sr)        # -> list of language regions

# Sources (async PCM-frame producers).
voh.sources.from_url(url)          # yt-dlp URL / RSS / direct audio   [stream]
voh.sources.from_microphone(...)   # live mic                          [mic]
voh.sources.from_numpy_array(pcm, sample_rate=..., frame_ms=20)
```

The public API is fixed via `vocal_helper.__all__`; treat those names as stable.

## 2. CLI — argparse (default) and click

`vocal-helper` (base) and `vocal-helper-click` (`[cli]` extra) — same
subcommands (`transcribe` / `file` / `url` / `mic`) and flags. See
`cli-reference.md`.

## 3. FastAPI HTTP surface (`[api]` extra)

```bash
pip install 'vocal-helper[api]'
uvicorn vocal_helper.api:app --host 0.0.0.0 --port 8000
```

| Method + path | Purpose |
|---------------|---------|
| `GET /health` | Liveness probe → `{"status": "ok"}`. |
| `GET /gui` | The transcript-viewer GUI (see §5). `/` redirects here. |
| `POST /transcribe` | One-shot ASR of an uploaded file → `{"text", "language"}`. |
| `POST /pipeline` | Full offline pipeline on an uploaded **file** *or* a **`url`** form field (fetched locally, needs `[stream]`) → `{"events": [...], "count"}`. |
| `GET /docs`, `GET /redoc` | OpenAPI docs. |

`/pipeline` events are `Utterance` dicts (`t0`, `t1`, `speaker`, `text`, …); a
`llm=true` run also emits `summary` snapshots. `diar_backend=auto` (default)
enforces the router server-side on the decoded buffer's real duration.

```bash
# File upload:
curl -F 'file=@meeting.wav' -F 'llm=true' http://localhost:8000/pipeline
# URL (fetched by the local server):
curl -F 'url=https://youtu.be/…' -F 'language=en' http://localhost:8000/pipeline
```

## 4. MCP surface (`[api,mcp]` extras)

```bash
pip install 'vocal-helper[api,mcp]'
vocal-helper-mcp                 # or: python -m vocal_helper.mcp
```

Wraps the FastAPI app with `fastapi-mcp`, publishing `transcribe` and `pipeline`
as MCP tools (same argument names as the HTTP routes) to any MCP-aware host.

## 5. Browser GUI — the transcript viewer (`GET /gui`)

A self-contained single page (HTML + Tailwind CDN + vanilla JS, no build step),
served same-origin by the API. Drop an audio file **or paste a URL**, run
diarized transcription locally, and read a **speaker-labelled, colour-coded
transcript** (one stable colour per speaker) plus the rolling summary. It POSTs
to the same `/pipeline` endpoint — zero extra server logic — and contacts only
the local server, so your audio never leaves the machine. Utterances reveal
progressively (motion-guarded) so a long transcript reads as if it streams in.

Open `http://localhost:8000/gui` (or just `http://localhost:8000/`, which
redirects there).
