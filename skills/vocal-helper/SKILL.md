---
name: vocal-helper
description: >-
  Turn speech — an audio file, a video's audio, a live microphone,
  or any yt-dlp-reachable URL — into a diarized, speaker-labelled transcript.
  Silero VAD → speaker diarization (torch-free Nemotron 3 ONNX by default;
  pyannote / NeMo / sherpa-onnx, auto-routed) → whisper.cpp / Nemotron STT → language
  ID → optional rolling local-LLM (Gemma) summary;
  offline or live. Use when the user says "transcribe this recording / meeting /
  interview / podcast", "give me the transcript", "make subtitles / captions /
  an SRT", "speech to text", "who spoke when", "diarize this", "label
  or separate the speakers", "how many speakers", "summarise this meeting
  audio", "what language is this", "live transcription from the mic", "diarize
  or transcribe with Nemotron / on GPU", or names `vocal-helper`, `vocal-helper-mcp`,
  `import vocal_helper as voh`. Do NOT use for audio-file transforms
  (audio-helper), text-to-speech, or downloading media for its own sake (youtube-helper). The
  exhaustive trigger catalogue is in `references/triggers.md`.
license: BSD-3-Clause
metadata:
  author: Warith Harchaoui
  package: vocal-helper
  suite: ai-helpers
  homepage: https://deraison.ai/ai-helpers
---

# vocal-helper — diarized transcription + rolling summary toolkit

`vocal-helper` is a local-first Python toolkit that turns speech audio into a
**diarized, speaker-labelled, transcribed** conversation, optionally with a
rolling LLM summary. It answers three questions at once: *what was said*
(Nemotron 3.5 ONNX or whisper.cpp STT), *who spoke when* (VAD + speaker diarization), and — with the
analyst stage — *what does it all mean* (local Gemma summary). The same pipeline
is reachable five ways (library, CLI, HTTP API, MCP, GUI) so an agent can
pick whichever fits. Nothing leaves the machine.

## Before anything: verify it is installed

```bash
vocal-helper --version            # or -V; the CLI ships with the package
python -c "import vocal_helper"   # library import check
```

If missing, install it (ffmpeg is a hard system dependency):

```bash
pip install vocal-helper                       # base: STT + torch-free diarization (Nemotron / sherpa ONNX), no compiler
pip install 'vocal-helper[pyannote]'           # + pyannote offline diarization (torch)
pip install 'vocal-helper[nemo]'               # + NeMo Sortformer diarization (~5 GB torch)
pip install 'vocal-helper[cloud]'              # + paid hosted modes (Gladia / ElevenLabs / pyannoteAI), key-gated
pip install 'vocal-helper[llm]'                # + local Gemma rolling summary (Ollama)
pip install 'vocal-helper[stream]'             # + URL ingest (YouTube / RSS / direct, yt-dlp)
pip install 'vocal-helper[mic]'                # + live microphone input
pip install 'vocal-helper[api,mcp]'            # + FastAPI HTTP, MCP tools (the CLI needs no extra)
pip install 'vocal-helper[all]'                # everything (heavy; pulls NeMo torch)
```

ffmpeg must be on PATH:
- macOS 🍎 : `brew install ffmpeg` (install `brew` via [brew.sh](https://brew.sh/))
- Ubuntu 🐧 : `sudo apt install ffmpeg`
- Windows 🪟 : `winget install Gyan.FFmpeg`

## What it does, and how to invoke it

| Intent | CLI | Library |
|--------|-----|---------|
| Transcribe an audio file / URL (words only) | `vocal-helper transcribe` | `transcribe_pcm_with_language` |
| Full offline pipeline — diarize + transcribe (+ summary) a file | `vocal-helper file` | `OfflinePipeline` |
| Full online pipeline on a URL (YouTube / RSS / direct) | `vocal-helper url <URL>` | `Pipeline` + `sources.from_url` |
| Live transcription from the microphone | `vocal-helper mic` | `Pipeline` + `sources.from_microphone` |
| Detect the spoken language / language regions | — | `detect_language`, `detect_language_regions` |

Quick examples:

```bash
# One-shot transcript of a file (auto language ID):
vocal-helper transcribe meeting.m4a

# Full diarized transcript with a rolling local-LLM summary, in French:
vocal-helper file interview.wav --language fr --llm

# Stream a YouTube talk through the pipeline (needs [stream]):
vocal-helper url "https://youtu.be/YE7VzlLtp-4" --language en

# Live mic (needs [mic]):
vocal-helper mic
```

```python
import vocal_helper as voh

# Full offline diarized pipeline over a decoded buffer.
pipe = voh.OfflinePipeline(source=my_frame_factory, config=cfg)
async for ev in pipe.run():
    # utterances carry t0/t1/speaker/text; summary snapshots carry `summary`
    print(ev.get("speaker"), ev.get("text") or ev.get("summary"))
```

The **diarization backend is auto-routed** by a study-grounded quality×speed
router: Nemotron 3 ONNX whenever its graph is in the engines bundle; without it,
short/dense audio (≤300 s, ≤4 speakers) → NeMo, long/unknown → pyannote, else
sherpa. Pass an explicit `--diar-backend nemotron|pyannote|nemo|sherpa` to override.

The **ASR engine is auto-picked** too (`--asr-engine auto`): Nemotron 3.5 ONNX when
the language is to be discovered (its 669 MB graph downloads once), whisper.cpp as
soon as `--language`, `--initial-prompt` or `--whisper-model` is given.
`--asr-engine whisper|nemotron` forces one.

For the full flag matrix and every option, read `references/cli-reference.md`.
For the API / MCP / GUI surfaces (endpoints, ports, the `/gui` transcript viewer),
read `references/surfaces.md`. For the exhaustive, auditable trigger list, read
`references/triggers.md`.

## Rules of thumb

- **Pick the surface from the intent.** Words only → `transcribe`. Who-spoke-when
  → `file` / `OfflinePipeline`. A URL → `url`. A microphone → `mic`. A browser
  view for a human → the `/gui` transcript viewer.
- **Video files and URLs are valid speech inputs.** ffmpeg decodes the audio
  track of `.mp4/.mkv/.mov/.webm`; `[stream]` fetches any yt-dlp URL — no need
  to pre-extract.
- **Language is discovered, not assumed.** `--language auto` (the default) lets
  the engine detect it; passing a code locks it and switches `auto` to whisper.
- **The base install diarizes.** sherpa-onnx and onnxruntime are core, so Nemotron
  (once the bundle is fetched) and sherpa run with no extra; `[pyannote]` and
  `[nemo]` add the torch backends.
- **The summary is opt-in and local.** `--llm` attaches the Gemma-via-Ollama
  analyst; it needs `[llm]` and a running local Ollama.
- **After running, report the transcript / output and the detected language and
  speakers** — do not re-run unless something failed.
- **Local by default.** No network except the one-time model downloads / yt-dlp
  fetch. The paid modes (`CloudOfflinePipeline`, `CloudLivePipeline`,
  `diar={"backend": "pyannoteai"}`) send audio out only when asked for by name
  with a key set; never pick them unless the user asks for a hosted service.
