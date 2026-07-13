# GUI — Vocal Helper

> A design plan, not a CLI mirror. The CLI already handles "point at a
> mic / file / URL and stream lines to stdout." A GUI must go further —
> otherwise why build one? This document lays out an ambitious,
> opinionated visual product for the *live-conversation intelligence*
> workflow.

## North star

> **A single window where a conversation is unfolding in real time,
> speaker-labelled, searchable, and augmented by a rolling LLM analyst
> that anyone in the room can steer.**

The CLI's strength is being pipeable to disk. It cannot show you *who
just spoke over whom*, *what the summary was 30 s ago*, or *how the
turn-taking dynamics have drifted*. That is what the GUI is for.

## Four surfaces, one product

### 1. Live Conversation Canvas *(primary surface)*

A vertical, ChatGPT-style transcript that grows downward — but with
per-speaker lanes (à la Google Meet captions) rather than a single
column. Each turn shows:

- Speaker label + colored dot (colorblind-safe via shape + text +
  color together — see companion `front-colors` audit skill).
- Timestamped bubble with the transcribed text; word-level highlight
  scrolls with playback if the audio is being replayed.
- Per-turn confidence bar (from whisper's per-token logprobs) —
  low-confidence turns get a faint yellow rail so you know where to
  double-check.
- Right-margin **"Ask about this"** button on hover: opens the LLM
  side-panel pre-seeded with the selected turn as context.

### 2. Rolling Summary Panel

Right-hand column, always visible. Two nested widgets:

- **Live summary** — updates every 60 s of evicted content. Rendered
  as a bulleted list; each bullet is clickable and jumps the transcript
  to the source turn that produced it (edge-of-summary provenance).
- **Recent verbatim window** — the last `recent_window_s` seconds
  verbatim, so you can see what has *not yet* been folded into the
  summary. This is the "what were we just saying?" widget users
  actually reach for.

### 3. Speaker Timeline

A horizontal timeline at the bottom, one row per detected speaker.
Bars represent voiced segments; bar height = per-window RMS. Purpose:

- **Turn-taking analytics.** Show the domination index (%
  floor-time), interruption rate, average pause length. Business
  users care about this in interviews, standups, therapy sessions,
  moderation.
- **Cluster reassignment.** Wrong speaker split? Drag a bar into a
  different lane → the diar model retraces from that segment forward
  with the corrected cluster hint. No CLI equivalent.
- **A/B backend comparison.** Toggle between `pyannote` and `nemo`
  online-diar backends on the same rendered audio; the timeline
  ghost-overlays both segmentations so an operator can pick.

### 4. Batch Meeting Vault

A dashboard listing every past session (as folders of PCM + JSONL).
Each row has a thumbnail speaker timeline, meeting duration, distinct
speaker count, and top-5 words (TF-IDF against the vault's own
distribution — not a global corpus). Full-text search across all
transcripts, filterable by speaker and date range.

## Design principles

- **Latency is visible.** The header shows a live "audio-arrived → text-emitted"
  ping counter. Users learn *why* an utterance is late (VAD hold,
  diar cold start, whisper backlog) instead of blaming "the AI".
- **Provenance is one click.** Every summary bullet links back to the
  transcript spans that produced it. Nothing is opaque.
- **Correction propagates.** Fixing a speaker label or a word retrains
  nothing — but it *does* patch the JSONL export, the summary, and
  the timeline, so downstream artifacts stay consistent.
- **Keyboard first, mouse second.** Space = play/pause, ←/→ = prev/next
  turn, J/K = prev/next low-confidence turn, /  = focus search, R =
  re-run diar with a fresh backend.
- **Local-only by default.** Every stage runs on the FastAPI server
  the container already ships. GUI is a thin JS client. No cloud.

## What we deliberately don't do

- **Not a transcription editor.** We show text; edits happen in an
  external tool (VSCode, Notion) via the JSONL export. Scope
  discipline — building a Word clone is a rabbit hole.
- **Not a video conferencing frontend.** We ingest audio; camera
  feeds are out of scope.
- **No cloud speech.** whisper.cpp on-device. Same-machine privacy.

## Stack

- Front end: TypeScript + Svelte 5 + WaveSurfer.js (spectrograms) +
  D3.js (timeline). No React — matches the `front-ui` companion skill
  stack.
- Back end: the FastAPI app already exists (`vocal_helper.api`) for
  offline processing. Live streaming uses a Server-Sent-Events /
  WebSocket adapter on top of the same `Pipeline` orchestrator.
- Session format: `pcm.wav` + `events.jsonl` (one Utterance /
  SummarySnapshot per line). Both artifacts are what the CLI already
  emits — the GUI does not invent a new schema.

## Milestones

| Milestone | What ships | Why first |
| --- | --- | --- |
| M0 | Live Conversation Canvas with mic input, 1 speaker, no summary. | Prove the low-latency read loop before scaling verbs. |
| M1 | Multi-speaker diarization + speaker lanes. | The first "obviously better than CLI" moment. |
| M2 | Rolling Summary Panel + clickable provenance. | The "why does this GUI exist" moment. |
| M3 | Speaker Timeline + drag-to-reassign. | Dataset-quality use case: interview coding, therapy notes. |
| M4 | Batch Meeting Vault + cross-session search. | Where the GUI passes the CLI in productivity. |
| M5 | A/B backend overlay + latency counter. | Operator observability — for the people who tune the stack. |

## Non-goals (recorded so we do not drift)

- Not a hosted SaaS.
- Not a full DAW.
- Not a substitute for the CLI in CI (batch mode calls the CLI
  directly — no GUI dependency in headless pipelines).

## Success metric

> A moderator running a live meeting closes the laptop at the end,
> re-opens it 3 h later, searches "action items", and finds every
> decision within 10 s with a click-to-audio jump.

If we ship that, we win.
