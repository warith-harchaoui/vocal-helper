---
name: vocal-helper
description: >-
  Turn audio (and the audio track of video, or any yt-dlp-reachable URL) into a
  diarized, speaker-labelled, transcribed conversation — plus an optional rolling
  local-LLM summary — with the `vocal-helper` toolkit. It answers "what was said",
  "who spoke when", and "who is who": Silero VAD → speaker diarization (pyannote /
  NeMo Sortformer / torch-free sherpa-onnx, auto-routed by a study-grounded quality
  ×speed router) → whisper.cpp speech-to-text → local language ID → optional
  Gemma-via-Ollama rolling summary. Two paths: an offline batch `OfflinePipeline`
  and an online streaming `Pipeline` (live mic / URL). Exposed as a Python library
  (`import vocal_helper as voh`), two CLIs (`vocal-helper` argparse and
  `vocal-helper-click`), a FastAPI HTTP surface, an MCP tool set, and a browser
  transcript-viewer GUI at `/gui` (drop a file or paste a URL → colour-coded,
  speaker-labelled transcript + summary). Local-first — whisper.cpp / pyannote /
  NeMo / sherpa / local Ollama; audio and transcripts never leave the machine, no
  telemetry, no account, no SaaS.

  TRIGGER — any of: the user asks to transcribe / caption / subtitle speech
  ("transcribe this recording / meeting / interview / podcast / lecture / voice
  memo", "turn this audio into text", "give me the transcript", "make subtitles /
  captions / an SRT / a VTT", "speech to text", "STT / ASR this"); the user asks
  who spoke ("who spoke when", "diarize this", "label the speakers", "separate the
  speakers", "speaker segmentation / turns", "how many speakers", "which speaker
  said X", "attribute each line to a speaker", "speaker-labelled transcript"); the
  user asks to identify speakers by name from the conversation; the user asks for a
  summary of a conversation / meeting ("summarise this meeting audio", "rolling
  summary", "minutes / notes from this recording", "TL;DR of this call"); the user
  asks the spoken language ("what language is this", "detect the language", "is
  this French or English", "which parts are in which language"); the user points at
  an audio file (`.wav .mp3 .m4a .m4b .flac .ogg .oga .opus .aac .wma .aiff`) or a
  video file whose speech is the target (`.mp4 .mkv .mov .webm .avi …`) OR a URL
  (YouTube / Vimeo / Twitch / SoundCloud / podcast RSS / direct audio) and wants
  the words / speakers / summary; the user wants live transcription of a microphone
  or a stream; the user types or references a command (`vocal-helper`,
  `vocal-helper-click`, `vocal-helper-mcp`, subcommands `mic|file|url|transcribe`)
  or a library symbol (`Pipeline`, `PipelineConfig`, `OfflinePipeline`,
  `OfflinePipelineConfig`, `transcribe_pcm`, `transcribe_pcm_with_language`,
  `SileroVADStage`, `OnlineDiarStage`, `OfflineDiarStage`, `WhisperStage`,
  `GemmaAnalystStage`, `select_diarization`, `detect_language`,
  `detect_language_regions`, `sources.from_url`, `sources.from_microphone`); the
  user wants the vocal-helper API / MCP server run, or the transcript-viewer GUI;
  the user asks to install / run vocal-helper.

  SKIP when: the task is a pure audio-file transform with no speech target —
  convert / re-encode / resample, cut / trim / split / concatenate, generate
  silence, room-tone, MFCC similarity, or Demucs stem separation (use audio-helper);
  text-to-speech / voice cloning / speech synthesis; music transcription (notes /
  MIDI, not speech); downloading a file from a URL for its own sake with no
  transcription wanted (use youtube-helper / podcast-helper); translating existing
  text (no audio in play); or non-speech audio classification (sound events, genre).
  vocal-helper turns *speech* into diarized, labelled, transcribed text (+ summary);
  it does not edit audio files, synthesize voices, or fetch media for storage.
---

# vocal-helper — diarized transcription + rolling summary toolkit

`vocal-helper` is a local-first Python toolkit that turns speech audio into a
**diarized, speaker-labelled, transcribed** conversation, optionally with a
rolling LLM summary. It answers three questions at once: *what was said*
(whisper.cpp STT), *who spoke when* (VAD + speaker diarization), and — with the
analyst stage — *what does it all mean* (local Gemma summary). The same pipeline
is reachable five ways (library, two CLIs, HTTP API, MCP, GUI) so an agent can
pick whichever fits. Nothing leaves the machine.

## Before anything: verify it is installed

```bash
vocal-helper --version            # argparse CLI (always installed with the pkg)
python -c "import vocal_helper"   # library import check
```

If missing, install it (ffmpeg is a hard system dependency):

```bash
pip install vocal-helper                       # base: STT + language ID (wheel-based, no compiler)
pip install 'vocal-helper[pyannote]'           # + robust offline diarization (torch)
pip install 'vocal-helper[sherpa]'             # + torch-free ONNX diarization (lightest)
pip install 'vocal-helper[nemo]'               # + NeMo Sortformer diarization (~5 GB torch)
pip install 'vocal-helper[llm]'                # + local Gemma rolling summary (Ollama)
pip install 'vocal-helper[stream]'             # + URL ingest (YouTube / RSS / direct, yt-dlp)
pip install 'vocal-helper[mic]'                # + live microphone input
pip install 'vocal-helper[cli,api,mcp]'        # + click CLI, FastAPI HTTP, MCP tools
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
router: short/dense audio (≤300 s, ≤4 speakers) → NeMo, long/unknown → pyannote,
torch-free installs → sherpa. Pass an explicit `--diar-backend` to override.

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
  whisper detect it; only pass a code when you must lock it.
- **Diarization needs a backend extra.** Base install transcribes but does not
  diarize; install `[pyannote]` (robust), `[sherpa]` (torch-free, lightest), or
  `[nemo]` (short-clip specialist). Absent → clear `ImportError` / 400.
- **The summary is opt-in and local.** `--llm` attaches the Gemma-via-Ollama
  analyst; it needs `[llm]` and a running local Ollama.
- **After running, report the transcript / output and the detected language and
  speakers** — do not re-run unless something failed.
- **Local only.** No network except the one-time model-bundle / yt-dlp fetch;
  audio and transcripts never go to a SaaS.
