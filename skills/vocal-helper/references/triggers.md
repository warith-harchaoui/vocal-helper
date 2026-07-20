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
  viewer / GUI", "install vocal-helper".

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
