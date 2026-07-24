# Landscape

[🇫🇷 PAYSAGE.md](https://github.com/warith-harchaoui/vocal-helper/blob/main/PAYSAGE.md) · 🇬🇧 English

Related and competing Python / OSS projects in the "live speech →
speaker-labelled text → summary" space, benchmarked against
`vocal-helper`. Ratings are ⭐ (1) to ⭐⭐⭐⭐⭐ (5), scored on
`vocal-helper`'s intended job — an **async producer/consumer pipeline
turning a live PCM stream (mic, URL, or file) into diarized,
transcribed utterances plus an optional rolling LLM summary**. A
project optimised for a different job (e.g. batch-only transcription,
non-streaming diarization, general-purpose LLM chat) is not penalised
— the score just reflects fit to *this* niche.

## At a glance

<!-- TABLE:START -->
| Live Transcription | Live streaming | Online diarization | Local-only STT | Rolling LLM summary | Multi-source | Ergonomic Python API | Multi-surface |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **vocal-helper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| pyannote.audio | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| NVIDIA NeMo | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| whisper.cpp | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| faster-whisper | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| whisper-live | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| RealtimeSTT | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| LiveKit Agents | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Pipecat | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| OpenAI Whisper | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| AssemblyAI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
<!-- TABLE:END -->

## Positioning map

<!-- FIGURE:START -->
2D representation of the table above.

![Positioning map](https://raw.githubusercontent.com/warith-harchaoui/vocal-helper/main/assets/landscape.png)

The map is a 2-D summary of the seven criteria, so read it as a shape, not a scoreboard. `vocal-helper` is at the top-right corner. The axes read **Horizontal — Self-reliant ↔ Integrated** and **Vertical — Contextual ↔ Versatile**.
<!-- FIGURE:END -->

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
notebook (no streaming), whisper.cpp (no diarization), or an agent
framework like LiveKit Agents / Pipecat (requires a lot of assembly
for local-only deployments).

Two nuances behind the stars are worth spelling out. Online
diarization is `vocal-helper`'s hardest constraint: it runs pyannote /
NeMo under an online clustering strategy, where `pyannote.audio`
itself scores highest offline but ships no default streaming pipeline.
The rolling LLM summary — Gemma via Ollama over a 60 s window — is a
built-in stage most ASR stacks simply do not have, which is why only
the agent frameworks (LiveKit Agents, Pipecat) come close on that
column.

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
