# Vocal Helper

[🇫🇷](LISEZMOI.md) · [🇬🇧](README.md)

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](#)

`Vocal Helper` belongs to a collection of libraries called `AI Helpers` developed for building Artificial Intelligence.

[🌍 AI Helpers](https://harchaoui.org/warith/ai-helpers)

Vocal Helper is an **async producer/consumer pipeline** turning a live PCM audio stream into diarized, transcribed utterances — and (optionally) a rolling LLM summary of the conversation.

## Pipeline

```
[Source]   →  [VAD]   →  [Online Diar]  →  [STT]   →  [LLM analyst (optional)]
  PCM         voiced     speaker-tagged     text          rolling summary
  frames      segments   segments
```

All edges are bounded `asyncio.Queue`s ; every stage is its own coroutine.

| Stage | Backend | Notes |
|---|---|---|
| **VAD** | Silero v5 ONNX (CPU) | 32 ms window, `activity_threshold=0.5`, default `min_silence_ms=300`. |
| **Diarization (online)** | `pyannote/embedding` (default) or `nvidia/titanet_large` (NeMo) | Per-segment embedding + cosine-distance running-mean clustering, `join_threshold=0.30`. Calibrated on AMI dev-slice N=8 (2026-06-30). |
| **STT** | [`pywhispercpp`](https://github.com/abdeladim-s/pywhispercpp) turbo | `large-v3-turbo-q5_0` by default. Word timestamps on. Runs in a thread pool so the event loop never stalls. |
| **LLM analyst** *(optional)* | Ollama-served Gemma 4 e4b (`gemma4:e4b`) | Rolling summary of everything **older than 60 s**. The recent 60 s window is kept verbatim. Apple-Silicon `-mlx` variant auto-selected by Ollama. |

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

### Replay a WAV through the pipeline

```bash
vocal-helper file path/to/conversation.wav --llm
```

The file source preserves real-time pacing by default ; pass `--no-real-time` for as-fast-as-possible batch processing.

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
