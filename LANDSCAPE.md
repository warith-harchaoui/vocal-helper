# Landscape

[🇫🇷 PAYSAGE.md](https://github.com/warith-harchaoui/vocal-helper/blob/main/PAYSAGE.md) · 🇬🇧 English

Related and competing Python / OSS projects in the "live speech →
speaker-labelled text → summary" space, benchmarked against
`vocal-helper`. Ratings are ⭐ (1) to ⭐⭐⭐⭐⭐ (5), scored on
`vocal-helper`'s intended job: an **async producer/consumer pipeline
turning a live PCM stream (mic, URL, or file) into diarized,
transcribed utterances plus an optional rolling LLM summary**. A
project optimised for a different job (e.g. batch-only transcription,
non-streaming diarization, general-purpose LLM chat) is not penalised;
the score just reflects fit to *this* niche.

## At a glance

<!-- TABLE:START -->
| Live Transcription | Live streaming | Online diarization | Local-only STT | Rolling LLM summary | Multi-source | Ergonomic Python API | Multi-surface | Offline |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **vocal-helper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| whisper.cpp | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| openai-whisper | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| faster-whisper | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| WhisperX | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| whisper_streaming | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| pyannote.audio | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| NVIDIA NeMo | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| SpeechBrain | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| vosk | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AssemblyAI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| Deepgram | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| Otter.ai | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐ |
<!-- TABLE:END -->

## Positioning map

<!-- FIGURE:START -->
2D representation of the table above.

![Positioning map](https://raw.githubusercontent.com/warith-harchaoui/vocal-helper/main/assets/landscape.png)

The map is a 2-D summary of the eight criteria, so read it as a shape, not a scoreboard. `vocal-helper` is at the top-right corner. The axes read **Horizontal: Resilient ↔ Dynamic** and **Vertical: Accessible ↔ Versatile**.
<!-- FIGURE:END -->

## Positioning

`vocal-helper` deliberately sits at the intersection of **whisper.cpp's
ergonomics** (local, cheap, no GPU strictly required) and the **live
diarization + rolling analyst** capability that most speech stacks
push off to the batch layer. It is not trying to beat `pyannote` on
offline DER or `faster-whisper` on raw ASR WER; it *composes* those
proven pieces into a single async pipeline whose stages are
individually swappable (any custom stage can be dropped in as a
coroutine), and it exposes the composition through four coherent
surfaces: argparse CLI, click CLI, FastAPI HTTP, MCP tools. That
trade-off is the main differentiator against a bare pyannote
notebook (no streaming), the local Whisper family (whisper.cpp,
openai-whisper, faster-whisper, WhisperX, whisper_streaming: ASR only,
no diarization), or a cloud service like AssemblyAI / Deepgram /
Otter.ai (fully managed, but off-device, scoring low on Local-only STT
and Offline).

Two nuances behind the stars are worth spelling out. Online
diarization is `vocal-helper`'s hardest constraint: it runs pyannote /
NeMo under an online clustering strategy, where `pyannote.audio`
itself scores highest offline but ships no default streaming pipeline.
The rolling LLM summary, a local model served over Ollama or vLLM and
resolved per machine by best-engine-ai-helper over a 60 s window, is a
built-in stage most ASR stacks simply do not have, which is why only
the cloud services (AssemblyAI, Deepgram) come close on that column,
and only by shipping your audio off-device.

## When to pick what

- **`vocal-helper`**: live conversation → diarized transcript →
  rolling summary, all on-device, embeddable in any Python service.
  Meetings, interviews, standups, therapy notes, moderation
  dashboards, voice-first agents.
- **`pyannote.audio`**: batch-only diarization on recorded audio
  where offline DER matters more than latency (podcast production,
  archive processing).
- **`NVIDIA NeMo`**: you already run a Triton / NIM stack and want
  Sortformer / TitaNet tightly coupled to your GPU serving layer.
- **`whisper.cpp` / `faster-whisper`**: you only need ASR, no
  diarization, no analyst; latency is not the tightest constraint.
- **`openai-whisper` / `WhisperX`**: you want the exact reference
  Whisper implementation for a benchmark (`openai-whisper`), or
  word-level alignment plus batch diarization on recordings
  (`WhisperX`); streaming is not on the table.
- **`whisper_streaming`**: you need a plug-and-play streaming ASR
  server without diarization or LLM stages.
- **`vosk` / `SpeechBrain`**: you want a small-footprint or
  research-friendly local ASR toolkit, no diarization analyst layer.
- **`AssemblyAI` / `Deepgram` / `Otter.ai`**: you accept cloud
  dependency (they score low on Local-only STT / Offline) and want a
  fully-managed SLA rather than a local pipeline.
