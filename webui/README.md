# webui — minimal local GUI for vocal-helper

A single, self-contained web page (no build step, no CDN, no external calls)
that drives the local vocal-helper pipeline through its FastAPI surface. Your
audio never leaves the machine — the page talks only to `127.0.0.1`.

Built following the [front-ui](https://) stack rules (semantic HTML, dark-mode
via `prefers-color-scheme`, visible focus rings, reduced-motion guard, the
three-Roboto typography with a system fallback so it works fully offline).

## Run it (one command)

```bash
pip install 'vocal-helper[api]'          # the FastAPI surface
uvicorn vocal_helper.api:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/ui** — the API serves this folder at `/ui`,
so the page is same-origin with the endpoints (no CORS to configure).

Prefer a truly zero-code UI? The API's Swagger page at
**http://127.0.0.1:8000/docs** already lets you upload a file and get a
transcript.

## What it does

- **Transcribe** — one-shot ASR (`POST /transcribe`); shows the text and the
  language *discovered* from the audio (language is never defaulted).
- **Full pipeline** — VAD → diarization → ASR (+ optional local Gemma summary)
  via `POST /pipeline`; shows speaker-labelled utterances. The diarization
  backend selector mirrors the [router](../README.md#backend-router--the-aiguilleur):
  `pyannote` (robust default), `nemo` (short ≤4-speaker clips), `sherpa`
  (torch-free ONNX).

## Files

- `index.html` — markup + self-contained styles (design tokens, both color
  schemes).
- `app.js` — a single ES module: builds the multipart request and renders the
  result. The only network call it makes is to your local server.
