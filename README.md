# Vocal Helper

[🇫🇷](LISEZMOI.md) · [🇬🇧](README.md)

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](#)

`Vocal Helper` belongs to a collection of libraries called `AI Helpers` developed for building Artificial Intelligence.

[🌍 AI Helpers](https://harchaoui.org/warith/ai-helpers)

[![logo](assets/logo.png)](https://harchaoui.org/warith/ai-helpers)

Vocal Helper is an **async producer/consumer pipeline** turning audio into diarized, transcribed utterances — and (optionally) a rolling LLM summary of the conversation. Two paths ship :

- **Online** (`vh.Pipeline`) — live PCM stream → live transcript + live summary. Each stage runs at its own cadence, decoupled by bounded queues.
- **Offline** (`vh.OfflinePipeline`) — full audio buffer → highest-quality diarization (pyannote 3.1 with the 300 s chunk-and-stitch path for long meetings) → transcript → summary.

## Pipelines

### Online (streaming)

```
[Source]   →  [VAD]   →  [Online Diar]  →  [STT]   →  [LLM analyst (optional)]
  PCM         voiced     speaker-tagged     text          rolling summary
  frames      segments   segments
```

### Offline (batch)

```
[Source]   →  [Offline Diar]  →  [STT]  →  [LLM analyst (optional)]
  full        full-buffer        text       rolling summary
  PCM         pyannote 3.1
              + chunk+stitch
              past 300 s
```

All edges are bounded `asyncio.Queue`s ; every stage is its own coroutine.

| Stage | Backend | Notes |
|---|---|---|
| **VAD** *(online only)* | Silero v5 ONNX (CPU) | 32 ms window, `activity_threshold=0.5`, default `min_silence_ms=300`. |
| **Online diarization** | `pyannote/embedding` (default) or `nvidia/titanet_large` (NeMo) | Per-segment embedding + cosine-distance running-mean clustering, `join_threshold=0.30`. Calibrated on AMI dev-slice N=8 (2026-06-30). |
| **Offline diarization** | `pyannote/speaker-diarization-3.1` (default) or `nvidia/diar_sortformer_v1` (NeMo) | Full-buffer call. Long inputs (> `ideal_duration_s` : **300 s** for pyannote, **60 s** for NeMo) are auto-chunked with 10 s overlap and stitched by cosine AHC at `stitch_threshold=0.35`. Constants codified in pdbms §10.6 (Bredin 2023 band, AMI dev-slice median DER 0.135). |
| **STT** | [`pywhispercpp`](https://github.com/abdeladim-s/pywhispercpp) turbo | `large-v3-turbo-q5_0` by default. Word timestamps on. Runs in a thread pool so the event loop never stalls. |
| **LLM analyst** *(optional)* | Ollama-served Gemma 4 e4b (`gemma4:e4b`) | Rolling summary of everything **older than 60 s**. The recent 60 s window is kept verbatim. Summary refreshes every **60 s of evicted content** (`flush_every_s=60`) — Pareto winner on cos_sim vs offline reference in `studies/llm_cadence_sweep.py` (2026-06-30 sweep, AMI IS1008a). Apple-Silicon `-mlx` variant auto-selected by Ollama. |

## Quickstart

### Install

```bash
pip install 'vocal-helper[all]'
```

The `[all]` extra brings the mic source, pyannote, and Ollama. Pick à la carte if you don't need everything :

| Extra | Brings | Required when |
|---|---|---|
| (none) | `pywhispercpp`, `silero-vad`, `audio-helper` | File / numpy sources, no diarization |
| `[mic]` | `capture-helper` | Live microphone source |
| `[pyannote]` | `pyannote.audio` | `diar={'backend': 'pyannote'}` (default) |
| `[nemo]` | `torch`, `nemo-toolkit[asr]` | `diar={'backend': 'nemo'}` |
| `[llm]` | `ollama` | `llm={'model': 'gemma4:e4b'}` |
| `[all]` | All of the above | One-line install |

You also need [Ollama](https://ollama.com) running locally if you enable the LLM analyst :

```bash
ollama pull gemma4:e4b
ollama serve   # usually launched at install time
```

### HuggingFace token

The pyannote backend pulls gated models from the Hub. `vocal-helper`
resolves the token in this order (first non-empty wins):

1. `--hf-token hf_…` on the CLI (or `hf_token=` kwarg on
   `OnlineDiarStage` / `OfflineDiarStage`).
2. The `HF_TOKEN` environment variable.
3. `secrets.hf_token` from a local `settings.yaml`.

To use the file path, copy the shipped template once and fill it in:

```bash
cp settings.yaml.example settings.yaml
# then edit `secrets.hf_token` — settings.yaml is git-ignored
```

The placeholder value `hf_XXXX` is treated as unset, so an unedited
copy never masquerades as real credentials.

### Live microphone → terminal

```bash
export HF_TOKEN=hf_yourtoken    # required to fetch pyannote/embedding
vocal-helper mic --llm
```

### Python API

```python
import asyncio
import vocal_helper as vh

async def main():
    pipeline = vh.Pipeline(
        source=lambda: vh.sources.from_microphone(),
        config=vh.PipelineConfig(
            diar={"backend": "pyannote"},
            asr={"model": "large-v3-turbo-q5_0", "language": "fr"},
            llm={"model": "gemma4:e4b"},   # remove to disable
        ),
    )
    async for ev in pipeline.run():
        if "text" in ev:
            print(f"[{ev['t0']:.1f} {ev['speaker']}] {ev['text']}")
        elif "summary" in ev:
            print(f"--- rolling summary ---\n{ev['summary']}")

asyncio.run(main())
```

### Replay a WAV through the **online** pipeline

```bash
vocal-helper file path/to/conversation.wav --llm
```

The file source preserves real-time pacing by default ; pass `--no-real-time` for as-fast-as-possible batch processing.

### **Offline** batch on a WAV (full-buffer pyannote 3.1)

```python
import asyncio, vocal_helper as vh

async def main():
    pipeline = vh.OfflinePipeline(
        source=lambda: vh.sources.from_wav_file(
            "meeting.wav", real_time=False
        ),
        config=vh.OfflinePipelineConfig(
            diar={"backend": "pyannote"},   # or "nemo" for ≤ 60 s clips
            asr={"language": "en"},
            llm={"model": "gemma4:e4b"},    # remove to disable
        ),
    )
    async for ev in pipeline.run():
        if "text" in ev:
            print(f"[{ev['t0']:.1f} {ev['speaker']}] {ev['text']}")
        elif "summary" in ev:
            print(f"--- digest ---\n{ev['summary']}")

asyncio.run(main())
```

When to use which :

| Use-case | Pipeline | Why |
|---|---|---|
| Live mic / live stream | `Pipeline` | Real-time diarization + transcript at RTF ≈ 0.03 (NeMo) or RTF ≈ 0.06 (pyannote). |
| Meeting recording / podcast / lecture / voicemail batch | `OfflinePipeline` | Pyannote 3.1 on the full audio is the highest-DER answer per the pdbms 2026-06-29 canonical study — median DER 0.116 on AMI dev-slice, inside Bredin 2023's band. |
| ≤ 60 s clips, fast turn-around | `OfflinePipeline(backend='nemo')` | NeMo Sortformer end-to-end attribution gives conf ≈ 0 ; RTF ≈ 0.004. |

## Multi-surface exposure

`vocal-helper` exposes the same pipeline through four coherent surfaces so it can slot into any host — a shell, a Python script, a container behind a reverse proxy, or an MCP-aware agent — without re-implementing the wiring anywhere else.

| Surface | Entry point | Extra | Kind of use |
|---|---|---|---|
| argparse CLI | `vocal-helper` | (none — ships with the base install) | Shell scripts, cron, headless CI, pipes to `jq`. |
| click CLI | `vocal-helper-click` | `[cli]` | Rich `--help`, shell completion, chained sub-commands. |
| FastAPI HTTP | `uvicorn vocal_helper.api:app` | `[api]` | Behind a reverse proxy — upload a file, get a transcript / a full event list, `GET /docs` for the OpenAPI. |
| MCP tools | `vocal-helper-mcp` | `[api,mcp]` | Any MCP-aware host — agent runtimes, IDE integrations — publishes `transcribe` and `pipeline` as first-class tools. |

```bash
# argparse
vocal-helper transcribe clip.wav --language en
vocal-helper file meeting.wav --offline --language en --llm

# click twin
vocal-helper-click transcribe clip.wav --language en

# HTTP surface
uvicorn vocal_helper.api:app --host 0.0.0.0 --port 8000 &
curl -F 'file=@clip.wav' -F 'language=en' http://localhost:8000/transcribe
curl -F 'file=@meeting.wav' -F 'llm=true'  http://localhost:8000/pipeline

# MCP surface (same FastAPI app + /mcp endpoint mounted)
vocal-helper-mcp
```

A one-line container recipe ships in `Dockerfile` — `docker build -t vocal-helper .` gets you an image serving both HTTP and MCP on `:8000`. See `GUI.md` for the (WIP) visual-product plan.

## Subscribers — fan-out without owning the loop

Every stage can be observed without consuming the merged output stream :

```python
async def on_voiced(seg): print("VAD:", seg["t0"], seg["t1"])
async def on_diar(seg):   print(" → ", seg["speaker"], seg["t0"], seg["t1"])

pipeline.subscribe_voiced(on_voiced)
pipeline.subscribe_diarized(on_diar)

async for ev in pipeline.run():
    ...
```

Useful for WebSocket / SSE relays, live UI updates, or JSONL persistence.

## Diarization choice — why **online cosine clustering**

The `pdbms` study (2026-06-29, N=2089 per system) ranks the online streaming diarizers as :

| Mode | Recommended | DER (clean) |
|---|---|---|
| Streaming ≤ 300 s | `hungarian_nemo` (w=20 s) | 0.13 – 0.20 |
| Streaming > 300 s | `hungarian_pyannote` (w=30 s) | 0.30 – 0.45 |

Vocal Helper specialises that decision : since the VAD already isolates each voiced segment for us, the sliding-window machinery collapses to per-segment embedding + cosine-distance running-mean clustering. The default `join_threshold=0.30` is the value selected on AMI dev-slice N=8 in the 2026-06-30 `pyannote_stitch_threshold_sweep`.

## Roadmap

- v0.2 — JSONL output writer + standard WebSocket relay (mirroring `capture-helper`'s publish path).
- v0.2 — language-locked Whisper rejection for ASR hallucinations on silence.
- v0.3 — anchor speaker IDs to enrolled voiceprints (carry over across sessions).
- v0.3 — replace the in-stage `_PyannoteEmbedder` with the overlap-aware variant from `pdbms.diar.backends.pyannote.embed_overlap_aware` for noisy mixes.

## Author

[Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui) — `warith@deraison.ai`
