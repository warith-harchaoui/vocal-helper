# LANDSCAPE

Related and competing Python / OSS projects in the "live speech →
speaker-labelled text → summary" space, benchmarked against
`vocal-helper`. Ratings are `⭐️` (1) to `⭐️⭐️⭐️⭐️⭐️` (5), scored on
`vocal-helper`'s intended job — an **async producer/consumer pipeline
turning a live PCM stream (mic, URL, or file) into diarized,
transcribed utterances plus an optional rolling LLM summary**. A
project optimised for a different job (e.g. batch-only transcription,
non-streaming diarization, general-purpose LLM chat) is not penalised
— the score just reflects fit to *this* niche.

## At a glance

| Library / project | Live streaming (frame-by-frame) | Online speaker diarization | Local-only STT (no cloud) | Rolling LLM summary (built-in) | Multi-source (mic + URL + file) | Ergonomic Python API (async, `dict` events) | Multi-surface exposure (CLI + HTTP + MCP) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **vocal-helper** *(this project)* | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ (pyannote / NeMo, online clustering) | ⭐️⭐️⭐️⭐️⭐️ (pywhispercpp turbo) | ⭐️⭐️⭐️⭐️ (Gemma via Ollama, 60 s window) | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ (argparse + click + FastAPI + MCP) |
| pyannote.audio | ⭐️⭐️ (streaming demos, no default pipeline) | ⭐️⭐️⭐️⭐️⭐️ (SotA offline, 3.x online mode) | n/a (diar only) | n/a | ⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️ |
| NVIDIA NeMo (ASR + Sortformer) | ⭐️⭐️⭐️ (streaming ASR ; Sortformer diar batch) | ⭐️⭐️⭐️⭐️ (Sortformer, batch by default) | ⭐️⭐️⭐️⭐️ | n/a | ⭐️⭐️⭐️ | ⭐️⭐️ (torch tensors, heavy) | ⭐️ |
| whisper.cpp (upstream) | ⭐️⭐️⭐️ (chunked-stream helper) | ⭐️ (no diar) | ⭐️⭐️⭐️⭐️⭐️ | ⭐️ | ⭐️⭐️ (CLI + file) | ⭐️⭐️ (C library) | ⭐️⭐️ (CLI-first) |
| faster-whisper | ⭐️⭐️⭐️ (chunk-loop patterns in the wild) | ⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️ | ⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️ |
| whisper-live | ⭐️⭐️⭐️⭐️ (WebSocket streaming server) | ⭐️ (optional pyannote adapter) | ⭐️⭐️⭐️⭐️⭐️ | ⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️ (WebSocket surface) |
| RealtimeSTT | ⭐️⭐️⭐️⭐️⭐️ (built for it) | ⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️ | ⭐️⭐️⭐️ (mic + file) | ⭐️⭐️⭐️⭐️ | ⭐️⭐️ |
| LiveKit Agents (voice) | ⭐️⭐️⭐️⭐️⭐️ (SFU-native) | ⭐️⭐️⭐️ | ⭐️⭐️⭐️ (multi-provider incl. local) | ⭐️⭐️⭐️⭐️⭐️ (agent framework) | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ (framework-native) |
| Pipecat | ⭐️⭐️⭐️⭐️ | ⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ (agent-shaped) | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ |
| OpenAI Whisper (upstream) | ⭐️ | ⭐️ | ⭐️⭐️⭐️⭐️⭐️ (but heavy) | ⭐️ | ⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️ |
| AssemblyAI (streaming) | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️ (cloud API) | ⭐️⭐️⭐️ (LeMUR) | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ (HTTP + WebSocket) |

## Positioning

`vocal-helper` deliberately sits at the intersection of **whisper.cpp's
ergonomics** (local, cheap, no GPU strictly required) and the **live
diarization + rolling analyst** capability that most speech stacks
push off to the batch layer. It is not trying to beat `pyannote` on
offline DER or `faster-whisper` on raw ASR WER — it *composes* those
proven pieces into a single async pipeline whose stages are
individually swappable (any custom stage can be dropped in as a
coroutine), and it exposes the composition through four coherent
surfaces: argparse CLI, click CLI, FastAPI HTTP, MCP tools. That
trade-off is the main differentiator against a bare pyannote
notebook (no streaming), whisper.cpp (no diar), or an agent framework
like LiveKit Agents / Pipecat (requires a lot of assembly for
local-only deployments).

## When to pick what

- **`vocal-helper`** — live conversation → diarized transcript →
  rolling summary, all on-device, embeddable in any Python service.
  Meetings, interviews, standups, therapy notes, moderation
  dashboards, voice-first agents.
- **`pyannote.audio`** — batch-only diarization on recorded audio
  where offline DER matters more than latency (podcast production,
  archive processing).
- **`NVIDIA NeMo`** — you already run a Triton / NIM stack and want
  Sortformer / TitaNet tightly coupled to your GPU serving layer.
- **`whisper.cpp` / `faster-whisper`** — you only need ASR, no
  diarization, no analyst; latency is not the tightest constraint.
- **`whisper-live` / `RealtimeSTT`** — you need a plug-and-play
  streaming ASR server without diarization or LLM stages.
- **`LiveKit Agents` / `Pipecat`** — you are building a voice AGENT
  (turn-based, TTS-out, tool-calling) and need SFU integration, not
  just an analyst.
- **`OpenAI Whisper` (upstream)** — you want the exact reference
  implementation for a benchmark; latency and streaming are not on
  the table.
- **`AssemblyAI` / hosted APIs** — you accept cloud dependency and
  want a fully-managed SLA rather than a local pipeline.
