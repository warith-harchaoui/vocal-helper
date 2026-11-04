# TRIGGERS — vocal-helper (exhaustive, auditable)

`vocal-helper` turns **speech** (audio, the audio track of video, or a URL) into
a **diarized, speaker-labelled, transcribed** conversation, optionally with a
rolling local-LLM summary, and can identify the spoken language. Local-first —
whisper.cpp / pyannote / NeMo / sherpa / local Ollama; nothing leaves the machine.

## What it does → how to invoke

| Intent | CLI | Library | API / MCP |
|--------|-----|---------|-----------|
| Transcribe speech (words only) | `vocal-helper transcribe` | `transcribe_pcm_with_language` | `POST /transcribe` |
| Diarize + transcribe a file (+ summary) | `vocal-helper file` | `OfflinePipeline` | `POST /pipeline` |
| Diarize + transcribe a URL | `vocal-helper url` | `Pipeline` + `sources.from_url` | `POST /pipeline` (`url` field) |
| Live mic transcription | `vocal-helper mic` | `Pipeline` + `sources.from_microphone` | — |
| Detect spoken language / regions | — | `detect_language`, `detect_language_regions` | — |
| Browser transcript viewer (for a human) | — | — | `GET /gui` |

**MCP** (`vocal-helper-mcp`, extra `[api,mcp]`): streamable-HTTP endpoint at
`http://127.0.0.1:8000/mcp` (`VOCAL_HELPER_HOST` / `VOCAL_HELPER_PORT`). The
tools are the API routes, under fastapi-mcp's operation ids:
`transcribe_transcribe_post`, `pipeline_pipeline_post`, `health_health_get`,
`gui_gui_get`. Same arguments as the HTTP routes.

## Engines and knobs (what to say to steer them)

| Ask | What it means here |
|-----|--------------------|
| "Nemotron", "Nemotron 3 Diarization", "Sortformer ONNX", "torch-free diarization", "diarize without HuggingFace" | the default offline diarizer: `NemotronOnnxDiar`, an ONNX graph from the diarization-engines bundle (no torch, no HF token) |
| "Nemotron 3.5 Transcribe", "transcribe with Nemotron", "torch-free / ONNX speech-to-text", "CTC timestamps", "transcribe without whisper" | `--asr-engine nemotron` (CLI), `asr={"engine": "nemotron"}` (library), `NemotronOnnxAsr`: the Nemotron 3.5 encoder + CTC head as one ONNX graph (no torch, no LLM, words timed at 40 ms, language discovered, speaker / event tags apart); graph from `VH_NEMOTRON_ASR_GRAPH` |
| "use pyannote / NeMo / sherpa instead" | `--diar-backend pyannote\|nemo\|sherpa` (CLI), `diar_backend` (API / MCP form field), `diar={"backend": …}` (library); `auto` lets the router pick |
| "run on GPU / CUDA / CoreML / DirectML / TensorRT / OpenVINO", "force CPU", "which ONNX Runtime provider" | `providers=` on `NemotronOnnxDiar`, or `VOCAL_HELPER_PROVIDER=cuda` (`cpu`, `coreml`, `directml`, `tensorrt`, `openvino`, `qnn`, `nnapi`) in the shell |
| "offline install", "air-gapped", "use my own engines bundle / mirror" | `VH_DIARIZATION_ENGINES` (URL or path of the bundle), `VH_DIARIZATION_ENGINES_SHA`, `VH_CACHE_DIR` |
| "low-latency live diarization", "label speakers faster" | `diar_latency_s=` (1.04 s or 0.32 s published schedules) |

## Natural-language phrasings that should fire

- **Transcribe / caption**: "transcribe this recording / meeting / interview /
  podcast / lecture / voice memo", "turn this audio into text", "give me the
  transcript", "make subtitles / captions / an SRT / a VTT", "speech to text",
  "STT / ASR this".
- **Diarize / who-spoke-when**: "who spoke when", "diarize this", "label / name
  the speakers", "separate the speakers", "speaker turns / segmentation", "how
  many speakers are there", "which speaker said X", "attribute each line to a
  speaker", "speaker-labelled transcript".
- **Summarise**: "summarise this meeting audio", "rolling summary", "minutes /
  notes from this recording", "TL;DR of this call".
- **Language ID**: "what language is this", "detect the language", "is this
  French or English", "which parts are in which language".
- **Live**: "transcribe my microphone", "live captions", "transcribe this
  stream / YouTube live as it plays".
- **Surfaces**: "run the vocal-helper API / MCP server", "open the transcript
  viewer / GUI", "install vocal-helper",
  "connect vocal-helper to my MCP client / Claude / IDE".
- **Engines**: "diarize with Nemotron", "torch-free / ONNX diarization",
  "run the diarizer on GPU / CUDA / CoreML", "pin the ONNX Runtime provider",
  "diarize offline / air-gapped", "transcribe with Nemotron",
  "ONNX speech-to-text without whisper or torch".

## File types & sources it accepts

- **Audio**: `.wav .mp3 .m4a .m4b .flac .ogg .oga .opus .aac .wma .aiff .aif`.
- **Video** (speech track decoded via ffmpeg): `.mp4 .mkv .mov .webm .avi …`.
- **URLs** (needs `[stream]`): YouTube / Vimeo / Twitch / SoundCloud, podcast
  RSS feeds (latest episode), and direct audio / HLS URLs — anything yt-dlp
  reaches.
- **Live**: the microphone (needs `[mic]`).

## When NOT to use vocal-helper (SKIP)

- Pure audio-file transforms with no speech target — convert / re-encode /
  resample, cut / trim / split / concatenate, silence, room-tone, MFCC
  similarity, or Demucs stem separation → use **audio-helper**.
- Text-to-speech, voice cloning, speech synthesis.
- Music transcription (notes / MIDI), not speech.
- Downloading media from a URL for storage with no transcription wanted → use
  **youtube-helper** / **podcast-helper**.
- Translating existing text (no audio in play).
- Non-speech audio classification (sound events, genre).

## See also

- [`../../README.md`](../../README.md) — features, install, quick start.
- [`../../TRIGGERS.md`](../../TRIGGERS.md) — the repo-level trigger catalogue.
- [`../README.md`](../README.md) — installing this as an agent skill.
- [`cli-reference.md`](cli-reference.md) · [`surfaces.md`](surfaces.md).
