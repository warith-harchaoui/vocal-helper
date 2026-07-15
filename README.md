# Vocal Helper

[🇫🇷](https://github.com/warith-harchaoui/vocal-helper/blob/main/LISEZMOI.md) · [🇬🇧](https://github.com/warith-harchaoui/vocal-helper/blob/main/README.md)

[![CI](https://github.com/warith-harchaoui/vocal-helper/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/warith-harchaoui/vocal-helper/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](#)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](.github/PULL_REQUEST_TEMPLATE.md)

`Vocal Helper` belongs to a collection of libraries called `AI Helpers` developed for building Artificial Intelligence.

[🌍 AI Helpers](https://harchaoui.org/warith/ai-helpers)

[![logo](https://raw.githubusercontent.com/warith-harchaoui/vocal-helper/main/assets/logo.png)](https://harchaoui.org/warith/ai-helpers)

Vocal Helper is an **async producer/consumer pipeline** turning audio into diarized, transcribed utterances — and (optionally) a rolling LLM summary of the conversation. Two paths ship :

- **Online** (`voh.Pipeline`) — live PCM stream → live transcript + live summary. Each stage runs at its own cadence, decoupled by bounded queues. The STT stage warms up on start so the first caption doesn't stall on whisper's cold inference.
- **Offline** (`voh.OfflinePipeline`) — full audio buffer → highest-quality diarization (pyannote 3.1 runs the whole meeting in one call — the 2026-07-14 offline map-reduce study found whole-buffer strictly best for DER; chunk-and-stitch survives only as a memory backstop past ~1 h) → **full-throttle batched transcript** (consecutive segments concatenated into ≤ 24 s whisper calls — ~6.5× lower RTF at better WER per the 2026-07-09 sweep) → summary. Opt back into per-segment ASR with `OfflinePipelineConfig(asr={"batch": False})`.

## Documentation

[💻 Documentation](https://harchaoui.org/warith/ai-helpers/docs/vocal-helper-doc/)

[📋 Examples](https://github.com/warith-harchaoui/vocal-helper/blob/main/EXAMPLES.md)

## Pipelines

Every edge is a bounded `asyncio.Queue` ; every stage is its own
coroutine. Colours follow the
[AI Helpers palette](https://harchaoui.org/warith/colors/).

### Online (streaming)

```mermaid
flowchart LR
    S([Source<br/><i>PCM frames</i>]):::source
      --> V[VAD<br/><i>Silero v5 ONNX</i>]:::vad
      --> D[Online Diar<br/><i>TitaNet · cosine clustering</i>]:::diar
      --> A[STT<br/><i>whisper.cpp turbo</i>]:::asr
      -.-> L[LLM analyst<br/><i>gemma3:4b · rolling summary</i>]:::llm

    classDef source fill:#CCE4FF,stroke:#007AFF,stroke-width:2px,color:#0b3d91
    classDef vad    fill:#00ffef,stroke:#79dbdc,stroke-width:2px,color:#003b3c
    classDef diar   fill:#EFDCF8,stroke:#AF52DE,stroke-width:2px,color:#4a1063
    classDef asr    fill:#FFEACC,stroke:#FF9500,stroke-width:2px,color:#5a3300
    classDef llm    fill:#D4F5D9,stroke:#28CD41,stroke-width:2px,color:#144d1e,stroke-dasharray: 5 5
```

The dashed edge marks the analyst as optional (`llm=None` disables it).

### Offline (batch)

```mermaid
flowchart LR
    S([Source<br/><i>full PCM buffer</i>]):::source
      --> D[Offline Diar<br/><i>pyannote 3.1<br/>whole-buffer</i>]:::diar
      --> A[STT<br/><i>whisper.cpp turbo</i>]:::asr
      -.-> L[LLM analyst<br/><i>gemma3:4b · rolling summary</i>]:::llm

    classDef source fill:#CCE4FF,stroke:#007AFF,stroke-width:2px,color:#0b3d91
    classDef diar   fill:#EFDCF8,stroke:#AF52DE,stroke-width:2px,color:#4a1063
    classDef asr    fill:#FFEACC,stroke:#FF9500,stroke-width:2px,color:#5a3300
    classDef llm    fill:#D4F5D9,stroke:#28CD41,stroke-width:2px,color:#144d1e,stroke-dasharray: 5 5
```

No VAD in the offline path — the diarizer consumes the whole buffer
and does its own segmentation.

| Stage | Backend | Notes |
|---|---|---|
| **VAD** *(online only)* | Silero v5 ONNX (CPU) | 32 ms window, `activity_threshold=0.5`, default `min_silence_ms=300`. |
| **Online diarization** | `nvidia/titanet_large` (NeMo, default) or `pyannote/embedding` | Per-segment embedding + cosine-distance running-mean clustering, `join_threshold=0.30`. Default backend switched to NeMo by the 2026-06-30 embedding sweep (`studies/diar_embedding_backend.py`): TitaNet has **+76 % separability margin** (inter − intra median cosine = 0.354 vs pyannote 0.201) on AMI dev-slice, at 7× per-call latency (45 ms vs 6 ms — still negligible per voiced segment). Pass `backend='pyannote'` to skip the ~ 5 GB NeMo install. |
| **Offline diarization** | `pyannote/speaker-diarization-3.1` (default) or `nvidia/diar_sortformer_v1` (NeMo) | Whole-buffer call. Inputs longer than `ideal_duration_s` (**3600 s** for pyannote — effectively whole-buffer, chunking is a memory backstop only; **60 s** for NeMo, forced by its Sortformer 90 s cap) are auto-chunked with 10 s overlap and stitched by cosine AHC at `stitch_threshold=0.35`. The 2026-07-14 offline map-reduce study found whole-buffer strictly best for DER (0.143 vs 0.170 at 300 s). |
| **STT** | [`pywhispercpp`](https://github.com/abdeladim-s/pywhispercpp) turbo | `large-v3-turbo-q5_0` by default. Word timestamps on. Runs in a thread pool so the event loop never stalls. **Strongly recommended: supply `initial_prompt` (domain bias)** — cuts WER 15-25 pp and saves up to 39 % RTF per the 2026-06-30 sweep (`studies/whisper_prompt_lang_lock.py`). |
| **LLM analyst** *(optional)* | Ollama-served Gemma 3 4b (`gemma3:4b`) | Rolling summary of everything **older than 60 s**. The recent 60 s window is kept verbatim. Summary refreshes every **60 s of evicted content** (`flush_every_s=60`). Default model `gemma3:4b` selected by the 2026-06-30 7-model Pareto sweep (`studies/llm_model_size_sweep.py`): it dominates `gemma4:e4b-mlx` on BOTH RTF (0.099 vs 0.313, **3× faster**) AND cos_sim (0.466 vs 0.420). Pareto front also exposes `gemma4:12b-mlx` (RTF 2.45, cos_sim 0.496) for offline-batch quality runs, and `qwen2.5:3b` (RTF 0.043) for tight RTF budgets. |

## Installation

> **More recipes?** See [`EXAMPLES.md`](https://github.com/warith-harchaoui/vocal-helper/blob/main/EXAMPLES.md)
> for a self-contained, copy-runnable cookbook of the common workflows
> (live mic, URL replay, offline batch, subscribers, multi-surface exposure).

> **Deploying on a GPU server?** See [TECHNICAL_STACK.md](https://github.com/warith-harchaoui/vocal-helper/blob/main/TECHNICAL_STACK.md)
> for the full stack recipe : CUDA + PyTorch, whisper.cpp with `GGML_CUDA=on`,
> pyannote 3.1 on MPS/CUDA, Ollama systemd, Docker Compose template,
> expected RTFs per GPU, and a 10-step reproducible install manifest
> covering the whole AI Helpers suite (os-helper, audio-helper,
> podcast-helper, youtube-helper, vocal-helper, music-helper).

**Prerequisites** — **Python 3.10–3.13** and **git**, **ffmpeg**, **PortAudio**, cross-platform:

- 🍎 **macOS** ([Homebrew](https://brew.sh)): `brew install python git ffmpeg portaudio`
- 🐧 **Ubuntu/Debian**: `sudo apt update && sudo apt install -y python3 python3-pip git ffmpeg portaudio19-dev`
- 🪟 **Windows** (PowerShell): `winget install Python.Python.3.12 Git.Git Gyan.FFmpeg` (PortAudio ships inside the Python wheels)

We recommend using Python environments. Check this link if you're unfamiliar with setting one up: [🥸 Tech tips](https://harchaoui.org/warith/4ml/#install).

### From PyPI (recommended)

```bash
pip install 'vocal-helper[all]'
```

### From source (no PyPI)

```bash
pip install 'vocal-helper[all] @ git+https://github.com/warith-harchaoui/vocal-helper.git@v0.4.3'
```

The `[all]` extra brings the mic source, both diarization backends (NeMo — the default — and pyannote), and Ollama. Pick à la carte if you don't need everything :

| Extra | Brings | Required when |
|---|---|---|
| (none) | `pywhispercpp`, `silero-vad`, `audio-helper` | File / numpy sources, no diarization |
| `[mic]` | `capture-helper` | Live microphone source |
| `[pyannote]` | `pyannote.audio` | `diar={'backend': 'pyannote'}` (lighter ~500 MB fallback) |
| `[nemo]` | `torch`, `nemo-toolkit[asr]` | `diar={'backend': 'nemo'}` (default — TitaNet, ~5 GB) |
| `[llm]` | `ollama` | `llm={'model': 'gemma3:4b'}` (default) |
| `[all]` | All of the above | One-line install |

You also need [Ollama](https://ollama.com) running locally if you enable the LLM analyst :

```bash
ollama pull gemma3:4b   # default (or gemma4:12b-mlx for max quality, qwen2.5:3b for min RTF)
ollama serve   # usually launched at install time
```

### Model weights — no HuggingFace needed

All model weights ship in a single self-hosted **diarization-engines
bundle** (offline pyannote 3.1, NeMo Sortformer, the online
`pyannote/embedding` embedder, SpeechBrain VoxLingua107). Point
`vocal-helper` at it once and the whole stack runs **HuggingFace-free** —
no token, no gated downloads, `HF_HUB_OFFLINE=1` safe.

Configure it in `settings.yaml` (the only config the project needs):

```bash
cp settings.yaml.example settings.yaml
# settings.yaml already contains:
#   engines:
#     diarization_url: https://deraison.ai/diarization-engines.zip
# settings.yaml is git-ignored.
```

#### What the URL is

`https://deraison.ai/diarization-engines.zip` is a **self-hosted ZIP**
(~800 MB) that mirrors every gated/hub-hosted model the pipeline needs, so
the project never has to authenticate against HuggingFace. It contains:

| Folder | Weights | Used by |
|---|---|---|
| `pyannote-3.1/` | segmentation-3.0 + wespeaker `.bin` + a local `config.yaml` | offline diarization |
| `nemo-sortformer/` | `diar_sortformer_4spk-v1.nemo` | offline diarization (NeMo) |
| `pyannote-embedding/` | embedding `.bin` | online diarization |
| `speechbrain-voxlingua107/` | ECAPA VoxLingua107 snapshot | language-ID cross-check |
| `manifest.json` | sha256 + sizes | integrity check on download |

On first use it is downloaded once, verified against `manifest.json`, and
cached under `~/.cache/vocal-helper`; later runs load straight from the
cache. Set `$VH_DIARIZATION_ENGINES` to a local directory (or your own
mirror URL) for air-gapped / self-hosted deploys. TitaNet (the default
online-diar embedder) loads from NVIDIA NGC, also without HuggingFace.

### Live microphone → terminal

```bash
# No token, no HuggingFace — weights come from the diarization-engines bundle.
vocal-helper mic --llm
```

### Python API

```python
import asyncio
import vocal_helper as voh

async def main():
    pipeline = voh.Pipeline(
        source=lambda: voh.sources.from_microphone(),
        config=voh.PipelineConfig(
            diar={"backend": "pyannote"},
            asr={"model": "large-v3-turbo-q5_0", "language": "fr"},
            llm={"model": "gemma3:4b"},   # remove to disable
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
import asyncio, vocal_helper as voh

async def main():
    pipeline = voh.OfflinePipeline(
        source=lambda: voh.sources.from_wav_file(
            "meeting.wav", real_time=False
        ),
        config=voh.OfflinePipelineConfig(
            diar={"backend": "pyannote"},   # or "nemo" for ≤ 60 s clips
            asr={"language": "en"},
            llm={"model": "gemma3:4b"},    # remove to disable
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

## Spoken-language identification

Before a word is transcribed, `vocal_helper.lid` decides **which language is
being spoken** — for the whole file, or per region of a code-switched
recording. This matters because a plain whisper `"auto"` pass locks onto the
first language it hears and *translates* the rest into it; identifying the
language acoustically **first** lets each region be transcribed in its own
language. It also catches mislabeled data: on a 423-call corpus the acoustic
census overrode the folder labels on 21 files (English and Dutch calls filed
under "FR", etc.).

| Function | What it does |
|---|---|
| `detect_language(pcm)` | One global detection, restricted to a routable `supported` set (so whisper never ranks an un-routable relative — Galician over Spanish — on a short window). Returns `(iso_639_1, probability)`. |
| `detect_language_regions(pcm)` | Partitions code-switched audio into mono-language `LangRegion`s via an overlapping-window **posterior curve** — Gaussian-smoothed, boundaries locally refined and snapped to the nearest silence. |
| `detect_language_regions_fast(pcm)` | Fast path *(new in 0.4.2)*: one cheap whole-file detection ; if it clears the confidence gate (`DEFAULT_FAST_CONF_GATE`, 0.5) on a routable language the file is treated as monolingual — a single region — otherwise it falls back to the full posterior scan. **~73 s → ~1 s per file** on the monolingual majority, identical output. |
| `cross_check_regions(pcm, regions)` | Optional independent verification with SpeechBrain VoxLingua107 (shipped in the diarization-engines bundle) — a second, model-diverse opinion on each region's language. |

```python
import vocal_helper as voh

# Fast path — the right default for batch corpora that are mostly monolingual:
regions = voh.detect_language_regions_fast(pcm, 16_000)
for r in regions:
    print(f"{r.lang}  [{r.t0:.1f}–{r.t1:.1f}s]")
```

The `supported` set defaults to a broad ISO-639-1 list ; restrict it to the
languages you can actually route (e.g. `supported=("en", "fr", "es", "it",
"pl", "nl")`) so a close but un-routable relative never wins.

## Roadmap

- v0.2 — JSONL output writer + standard WebSocket relay (mirroring `capture-helper`'s publish path).
- v0.2 — language-locked Whisper rejection for ASR hallucinations on silence.
- v0.2 — `SemanticEOTStage` enabled-by-default after the 2026-06-30 EOT study validates the false-cut reduction on AMI (LiveKit-style ; see `studies/eot_semantic_vs_silero.py`).
- v0.2 — auto LLM-engine selector : Ollama+MLX on Apple Silicon, vLLM on Linux+NVIDIA, llama.cpp gguf on CPU fallback.
- v0.3 — out of scope : speaker ID anchoring via pre-enrolled voiceprints (excluded by user's industrial deployment compliance constraints — IDs stay anonymous `S0`, `S1`, … within a session).
- v0.3 — replace the in-stage `_PyannoteEmbedder` with the overlap-aware variant from `pdbms.diar.backends.pyannote.embed_overlap_aware` for noisy mixes.
- v0.3 — Pipecat-style typed Frame events with SystemFrame priority queue (clean shutdown / out-of-band control signals that bypass DataFrame queues).

## Author

[Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui) — `warith@deraison.ai`

## Acknowledgements

Special thanks to
[Mohamed Chelali](https://mchelali.github.io),
[Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug)
and
[Edmond Jacoupeau](https://www.crunchbase.com/person/edmond-jacoupeau).

## License

This project is licensed under the BSD-3-Clause License — see the [LICENSE](LICENSE) file for details.
