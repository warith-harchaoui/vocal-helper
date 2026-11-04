# TRIGGERS: vocal-helper

This is the user-facing, exhaustive catalogue of what `vocal-helper` can do and
the natural-language phrasings, commands, functions, file types, and URLs that
should invoke it, whether you call it yourself or drive it as a Claude /
OpenCode **skill** (see [`skills/vocal-helper/SKILL.md`](skills/vocal-helper/SKILL.md)
and its [`references/triggers.md`](skills/vocal-helper/references/triggers.md)).

`vocal-helper` turns **speech** (audio, the audio track of video, or any
yt-dlp-reachable URL) into a **diarized, speaker-labelled, transcribed**
conversation, with an optional rolling local-LLM summary. It also identifies
the spoken language. It is **local-first**: whisper.cpp / pyannote / NeMo /
sherpa / local Ollama, no telemetry, no account, no SaaS. It does **not** edit
audio files, synthesize voices, or fetch media just for storage.

## What it does → how to invoke

| Intent | CLI | Library | API / MCP |
|--------|-----|---------|-----------|
| Transcribe speech (words only) | `vocal-helper transcribe` | `transcribe_pcm_with_language` | `POST /transcribe` |
| Diarize + transcribe a file (+ summary) | `vocal-helper file` | `OfflinePipeline` | `POST /pipeline` |
| Diarize + transcribe a URL | `vocal-helper url` | `Pipeline` + `sources.from_url` | `POST /pipeline` (`url` field) |
| Live mic transcription | `vocal-helper mic` | `Pipeline` + `sources.from_microphone` | — |
| Detect spoken language / regions | — | `detect_language`, `detect_language_regions` | — |

Every subcommand is also reachable through the MCP tools (`vocal-helper-mcp`)
and the browser **transcript viewer** at `GET /gui` (drop a file or paste a URL
→ colour-coded, speaker-labelled transcript + summary).

**MCP** (`vocal-helper-mcp`, extra `[api,mcp]`): streamable-HTTP endpoint at
`http://127.0.0.1:8000/mcp` (`VOCAL_HELPER_HOST` / `VOCAL_HELPER_PORT`). The
tools are the API routes, under fastapi-mcp's operation ids:
`transcribe_transcribe_post`, `pipeline_pipeline_post`, `health_health_get`,
`gui_gui_get`. Same arguments as the HTTP routes.

## Engines and knobs (what to say to steer them)

| Ask | What it means here |
|-----|--------------------|
| "Nemotron", "Nemotron 3 Diarization", "Sortformer ONNX", "torch-free diarization", "diarize without HuggingFace" | the default offline diarizer: `NemotronOnnxDiar`, an ONNX graph from the diarization-engines bundle (no torch, no HF token) |
| "use pyannote / NeMo / sherpa instead" | `--diar-backend pyannote\|nemo\|sherpa` (CLI), `diar_backend` (API / MCP form field), `diar={"backend": …}` (library); `auto` lets the router pick |
| "run on GPU / CUDA / CoreML / DirectML / TensorRT / OpenVINO", "force CPU", "which ONNX Runtime provider" | `providers=` on `NemotronOnnxDiar`, or `VOCAL_HELPER_PROVIDER=cuda` (`cpu`, `coreml`, `directml`, `tensorrt`, `openvino`, `qnn`, `nnapi`) in the shell |
| "offline install", "air-gapped", "use my own engines bundle / mirror" | `VH_DIARIZATION_ENGINES` (URL or path of the bundle), `VH_DIARIZATION_ENGINES_SHA`, `VH_CACHE_DIR` |
| "low-latency live diarization", "label speakers faster" | `diar_latency_s=` (1.04 s or 0.32 s published schedules) |

## Natural-language phrasings that should fire

- **Transcribe / caption**: "transcribe this recording / meeting / interview /
  podcast / lecture / voice memo", "turn this audio into text", "give me the
  transcript", "make subtitles / captions / SRT / VTT", "speech to text",
  "STT / ASR this".
- **Diarize / who-spoke-when**: "who spoke when", "diarize this", "label / name
  the speakers", "separate the speakers", "speaker turns / segmentation", "how
  many speakers", "which speaker said X", "speaker-labelled transcript".
- **Summarise**: "summarise this meeting audio", "rolling summary", "minutes /
  notes from this recording", "TL;DR of this call".
- **Language ID**: "what language is this", "detect the language", "is this
  French or English", "which parts are in which language".
- **Live**: "transcribe my microphone", "live captions", "transcribe this stream
  as it plays".
- **Surfaces**: "run the vocal-helper API / MCP server", "open the transcript
  viewer / GUI", "install vocal-helper",
  "connect vocal-helper to my MCP client / Claude / IDE".
- **Engines**: "diarize with Nemotron", "torch-free / ONNX diarization",
  "run the diarizer on GPU / CUDA / CoreML", "pin the ONNX Runtime provider",
  "diarize offline / air-gapped".

## Sources it accepts

- **Audio**: `.wav .mp3 .m4a .m4b .flac .ogg .oga .opus .aac .wma .aiff .aif`.
- **Video** (speech track decoded via ffmpeg): `.mp4 .mkv .mov .webm .avi …`.
- **URLs** *(needs `[stream]`)*: YouTube / Vimeo / Twitch / SoundCloud, podcast
  RSS (latest episode), direct audio / HLS: anything yt-dlp reaches.
- **Live**: the microphone *(needs `[mic]`)*.

## When NOT to use vocal-helper (SKIP)

- Pure audio-file transforms with no speech target: convert / re-encode /
  resample, cut / trim / split / concatenate, silence, room-tone, MFCC
  similarity, Demucs stem separation → use **audio-helper**.
- Text-to-speech, voice cloning, speech synthesis.
- Music transcription (notes / MIDI), not speech.
- Downloading media from a URL for storage with no transcription wanted → use
  **youtube-helper** / **podcast-helper**.
- Translating existing text (no audio in play).
- Non-speech audio classification (sound events, genre).

## See also

- [`README.md`](README.md): features, install, quick start.
- [`LISEZMOI.md`](LISEZMOI.md): version française.
- [`EXAMPLES.md`](EXAMPLES.md): runnable recipes.
- [`skills/README.md`](skills/README.md): installing this as an agent skill.
