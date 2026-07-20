# vocal-helper CLI reference

Two CLIs ship the same subcommands and flags: `vocal-helper` (stdlib argparse,
always installed) and `vocal-helper-click` (needs the `[cli]` extra). Use either.

```bash
vocal-helper --help
vocal-helper <subcommand> --help
```

## Subcommands

| Subcommand | Purpose | Extra needed |
|------------|---------|--------------|
| `transcribe` | One-shot speech-to-text of a file or URL (no diarization) | base |
| `file`       | Full pipeline on a file — VAD + diarization + STT (+ summary) | a diar backend |
| `url`        | Full pipeline on any yt-dlp URL (YouTube / RSS / direct audio) | `[stream]` + diar |
| `mic`        | Live transcription from the microphone | `[mic]` + diar |

## Common flags (on `file` / `url` / `mic` unless noted)

| Flag | Default | Meaning |
|------|---------|---------|
| `--language` | `auto` | ISO-639-1 code or `auto` (whisper detects it). |
| `--whisper-model` | `large-v3-turbo-q5_0` | whisper.cpp model id. |
| `--threads` | `6` | whisper.cpp decode threads. |
| `--diar-backend` | `auto` | `auto` \| `pyannote` \| `nemo` \| `sherpa`. `auto` routes by length. |
| `--llm` | off | Attach the local Gemma rolling-summary analyst (needs `[llm]`). |
| `--llm-model` | `gemma4:e4b` | Ollama model for the summary. |
| `--llm-recent-window-s` | `60` | Verbatim recent-transcript window before the digest. |

`file`-only levers cover offline batching (see `vocal-helper file --help`).

## The diarization router (the *aiguilleur*)

`--diar-backend auto` (default) hands the file's real duration to
`select_diarization`, trading quality (DER) against speed (RTF) per scenario:

- offline short (≤300 s, ≤4 speakers) → **nemo** (best DER on short/dense audio)
- offline long / unknown length → **pyannote** (robust default)
- torch-free install → **sherpa** (ONNX, no PyTorch)
- online (mic / url) → **nemo**

It never routes to an uninstalled backend — a short file with no `[nemo]` extra
falls back to pyannote rather than crashing. Any explicit `--diar-backend`
overrides the router verbatim.

## Examples

```bash
# Words only, auto language:
vocal-helper transcribe voice-memo.m4a

# Diarized transcript + rolling French summary:
vocal-helper file interview.wav --language fr --llm

# Pin the torch-free backend explicitly:
vocal-helper file meeting.mp3 --diar-backend sherpa

# Stream a podcast RSS (latest episode) through the pipeline:
vocal-helper url "https://feeds.example.com/show.xml" --language en

# Live microphone:
vocal-helper mic --diar-backend nemo
```

## Output contract

Utterances print as `[t0s speaker] text`, one per turn, in time order; a
`--llm` run also prints rolling `— summary —` snapshots. The detected language
is reported so callers never have to assume it. For machine-readable JSON, use
the HTTP `/pipeline` endpoint (see `surfaces.md`).
