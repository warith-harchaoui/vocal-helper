# TRIGGERS: vocal-helper

This is the user-facing, exhaustive catalogue of what `vocal-helper` can do and
the natural-language phrasings, commands, functions, file types, and URLs that
should invoke it, whether you call it yourself or drive it as a Claude /
OpenCode **skill**. The `description:` in `skills/vocal-helper/SKILL.md` is what
a host model actually sees and is capped at 1024 characters; this file carries
the full list, so keep the two in sync. The repo-root `TRIGGERS.md` and
`skills/vocal-helper/references/triggers.md` are the same file, byte for byte;
`scripts/check_triggers.py` fails when they drift.

`vocal-helper` turns **speech** (audio, the audio track of video, or any
yt-dlp-reachable URL) into a **diarized, speaker-labelled, transcribed**
conversation, with an optional rolling local-LLM summary. It also identifies
the spoken language. It is **local-first**: Nemotron ONNX / whisper.cpp / pyannote / NeMo /
sherpa / local Ollama, no telemetry, no account, no SaaS unless a paid
`[cloud]` mode is asked for by name with its key. It does **not** edit
audio files, synthesize voices, or fetch media just for storage.

## What it does → how to invoke

| Intent | CLI | Library | API / MCP |
|--------|-----|---------|-----------|
| Transcribe speech (words only) | `vocal-helper transcribe` | `transcribe_pcm_with_language` | `POST /transcribe` |
| Diarize + transcribe a file (+ summary) | `vocal-helper file` | `OfflinePipeline` | `POST /pipeline` |
| Diarize + transcribe a URL | `vocal-helper url` | `Pipeline` + `sources.from_url` | `POST /pipeline` (`url` field) |
| Live mic transcription | `vocal-helper mic` | `Pipeline` + `sources.from_microphone` | — |
| Detect spoken language / regions | — | `detect_language`, `detect_language_regions` | — |
| Browser transcript viewer (for a human) | — | — | `GET /gui` |

The file, URL and one-shot transcription paths are also reachable through the
MCP tools (`vocal-helper-mcp`) and the browser **transcript viewer** at
`GET /gui` (drop a file or paste a URL → colour-coded, speaker-labelled
transcript + summary). The live microphone and the language-ID functions have
no HTTP route, so no MCP tool either.

**MCP** (`vocal-helper-mcp`, extra `[api,mcp]`): streamable-HTTP endpoint at
`http://127.0.0.1:8000/mcp`. The server binds `0.0.0.0:8000` by default;
`--host` / `--port` or `VOCAL_HELPER_HOST` / `VOCAL_HELPER_PORT` change it,
and `vocal-helper-mcp --help` lists them. The tools are the API routes, under
fastapi-mcp's operation ids:
`transcribe_transcribe_post`, `pipeline_pipeline_post`, `health_health_get`,
`gui_gui_get`. Same arguments as the HTTP routes.

## Engines and knobs (what to say to steer them)

| Ask | What it means here |
|-----|--------------------|
| "Nemotron", "Nemotron 3 Diarization", "Sortformer ONNX", "torch-free diarization", "diarize without HuggingFace" | the default diarizer, offline and live, when its graph is in the bundle: `NemotronOnnxDiar`, an ONNX graph from the diarization-engines bundle (no torch, no HF token); `--diar-backend nemotron` forces it |
| "use Gladia / ElevenLabs Scribe / pyannoteAI", "hosted / paid transcription", "cloud diarization" | needs `[cloud]` plus the provider key (`GLADIA_API_KEY`, `ELEVENLABS_API_KEY`, `PYANNOTEAI_API_KEY`): `CloudOfflinePipeline`, `CloudLivePipeline`, or the hybrid `diar={"backend": "pyannoteai"}` with local words; `credentials_status()` says which are configured. Never picked by the router, never without being asked |
| "Nemotron 3.5 Transcribe", "transcribe with Nemotron", "torch-free / ONNX speech-to-text", "CTC timestamps", "transcribe without whisper" | `--asr-engine auto` (default: Nemotron when the language is to be discovered, offline and live; whisper when `--language` / `--initial-prompt` / `--whisper-model` is given) or `nemotron` (CLI), `asr={"engine": "nemotron"}` (library), `NemotronOnnxAsr`: the Nemotron 3.5 encoder + CTC head as one ONNX graph (no torch, no LLM, words timed at 40 ms, language discovered, speaker / event tags apart); the int8 graph is fetched once from `deraison.ai/nemotron-onnx` (`precision="fp32"` for the exact one), or read from `VH_NEMOTRON_ASR_GRAPH`; mirror with `VH_NEMOTRON_ASR_URL` |
| "use pyannote / NeMo / sherpa instead" | `--diar-backend nemotron\|pyannote\|nemo\|sherpa` (CLI), `diar_backend` (API / MCP form field), `diar={"backend": …}` (library); `auto` lets the router pick |
| "run on GPU / CUDA / CoreML / DirectML / TensorRT / OpenVINO", "force CPU", "which ONNX Runtime provider" | `providers=` on `NemotronOnnxDiar`, or `VOCAL_HELPER_PROVIDER=cuda` (`cpu`, `coreml`, `directml`, `tensorrt`, `openvino`, `qnn`, `nnapi`) in the shell |
| "offline install", "air-gapped", "use my own engines bundle / mirror" | `VH_DIARIZATION_ENGINES` (URL or path of the bundle), `VH_DIARIZATION_ENGINES_SHA`, `VH_NEMOTRON_ASR_GRAPH` / `VH_NEMOTRON_ASR_URL` (ASR graph), `VH_CACHE_DIR` |
| "low-latency live diarization", "label speakers faster" | `diar_latency_s=` (1.04 s or 0.32 s published schedules) |

## Natural-language phrasings that should fire

- **Transcribe / caption**: "transcribe this recording / meeting / interview /
  podcast / lecture / voice memo", "turn this audio into text", "give me the
  transcript", "make subtitles / captions / an SRT / a VTT", "speech to text",
  "STT / ASR this". vocal-helper returns timed, speaker-labelled utterances;
  it writes no `.srt` / `.vtt` file itself, so the caller formats them.
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

- the repo's `README.md`: features, install, quick start.
- the repo's `LISEZMOI.md`: version française.
- the repo's `EXAMPLES.md`: runnable recipes.
- the repo's `skills/README.md`: installing this as an agent skill.
- the skill's `references/cli-reference.md` and `references/surfaces.md`.
