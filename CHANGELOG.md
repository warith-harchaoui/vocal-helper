# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org). Pre-1.0
stability policy: **breaking behaviour and default changes land only in MINOR
releases; PATCH releases are bug-fixes and docs only.** The public API is the
names exported from `vocal_helper.__all__` plus the documented CLI flags.

## [0.7.1] - 2026-07-23

### Added

- **Router: a known-speaker-count prior.** `select_diarization` takes a new optional
  `num_speakers` (the exact count, distinct from the `max_speakers` ceiling). When the
  router picks `sherpa`, the returned `BackendPlan` now carries `sherpa_num_clusters`
  so the consumer can pin sherpa's clustering to that count, the lever that collapses
  sherpa's 2-party telephony over-segmentation (pdbms diar-study §12.1). Leaving
  `num_speakers` unset keeps sherpa on auto-clustering, so meeting-audio behaviour is
  unchanged. Additive and backward-compatible.
- `BackendPlan.rationale`, a read-only alias for `reason` (some consumers log the
  former; it was previously always empty).

## [0.7.0] - 2026-07-23

### Added

- **`OfflineDiarStage` now exposes sherpa's clustering knobs.** Two new
  backward-compatible parameters, `sherpa_cluster_threshold` (default `0.5`) and
  `sherpa_num_clusters` (default `-1`, auto), are forwarded into sherpa-onnx's
  `FastClusteringConfig`. They were previously hardcoded, which made
  `stitch_threshold` a silent no-op on the sherpa backend and left its clustering
  untunable.

### Fixed

- **Sherpa over-segmentation on noisy telephony is now controllable.** The
  hardcoded threshold `0.5` (tuned on clean AMI meeting audio) over-segments real,
  noisy, Personally Identifiable Information (PII)-redacted phone calls into dozens of speakers (we measured ~36 on
  2-party calls). A 2026-07-23 sweep against a pyannoteAI silver ground truth showed
  that raising `sherpa_cluster_threshold` helps only slowly (still ~30 speakers at
  0.6). When the speaker count is known, for instance 2-party telephony, setting
  `sherpa_num_clusters=2` brings it back down to the right count. Defaults are
  unchanged, so existing meeting-audio behaviour is preserved.

## [0.6.1] - 2026-07-21

### Fixed

- **Analyst time cadence now fires from a cold start.** `_on_utterance`
  measured the pending span with `newest_t1 - (self._oldest_pending_t0 or
  newest_t1)`, so a legitimate oldest-`t0` of `0.0` — every session's very first
  utterance — was treated as unset, collapsing the span to `0` and preventing
  `flush_every_s` from ever tripping until the queue rolled past that first
  utterance. Aligned the read with an explicit `is None` check (matching the
  sibling eviction guard). Regression test starts timestamps at `t0=0.0` and
  asserts the cadence still fires.

### Internal

- Error-path and edge-case test coverage for the analyst stage (prompt
  assembly + block fold, empty-queue no-op, Large Language Model (LLM)-failure keeps summary / drops
  poisoned block, Voice Activity Detection (VAD)-blip skip, time-vs-count cadence selection, run() drain +
  shutdown flush, missing-`[llm]` extra ImportError) plus VAD, sources, Language Identification (LID),
  API, and CLI. Coverage floor raised 35 → 55.
- Local CI-mirror gate: `make preflight` (pre-commit + ruff + tests) wired as a
  `pre-push` hook via `make install-hooks`.

## [0.6.0] - 2026-07-20

### Added

- **Transcript-viewer GUI at `GET /gui`.** A self-contained single page
  (HTML + Tailwind CDN + vanilla JS, no build step) served same-origin by the
  FastAPI app from the new in-package `vocal_helper.gui` module (mirrors the
  AI Helpers suite convention, e.g. `audio_helper.gui`). Drop an audio file
  **or paste a URL** → run the offline diarized pipeline → read a
  **speaker-labelled, colour-coded transcript** (one stable colour per speaker,
  with a live legend) alongside the rolling Gemma summary. Utterances reveal
  progressively (motion-guarded) so a long transcript reads as if it streams in.
  It POSTs to the existing `/pipeline` endpoint — zero new server logic — and
  contacts only the local server, so audio never leaves the machine. `GET /`
  now redirects to `/gui`. Because the module ships inside the package, the GUI
  works for a bare `pip install`, not only a source checkout.
- **URL ingest on `POST /pipeline`.** The endpoint now accepts an optional `url`
  form field (in addition to a file upload): the LOCAL server fetches any
  yt-dlp-reachable URL / RSS (Really Simple Syndication) / direct audio via `sources.from_url` (needs the
  `[stream]` extra). This powers the GUI's "paste a URL" affordance end-to-end.
  The uploaded `file` becomes optional; exactly one of `file` / `url` is required
  (a clean 400 otherwise).
- **Agent skill family `skills/vocal-helper/`** — a Claude Skill *and* an
  OpenCode skill (`SKILL.md` with a trigger-exhaustive third-person description
  + `references/{cli-reference,surfaces,triggers}.md`) plus `skills/README.md`
  with symlink-install instructions for `~/.claude/skills/` and
  `~/.opencode/skills/`.
- **`TRIGGERS.md`** — the exhaustive, user-facing catalogue of what invokes
  vocal-helper (transcription / diarization / "who spoke when" / summary /
  subtitles / language ID) and its SKIP boundary, referenced from README +
  LISEZMOI.
- **Local-first badge + `## The Promise` / `## La promesse`** sections in
  README / LISEZMOI, emphasising that voice and transcripts stay on-device.

### Changed

- The older static form GUI at `/ui` (repo `webui/` staticfiles) is kept for
  backward compatibility; `/gui` is now the canonical, in-package GUI.
- **`os-helper` pin lowered `>=1.5.3` → `>=1.5.2`** (the published floor). The
  diarization-bundle fetch prefers `os_helper.download_file` when present and
  falls back to a stdlib streaming download on older os-helper, so the pin stays
  satisfiable against PyPI without depending on an unpublished sibling release.

### Notes

- Public API unchanged — every name in `vocal_helper.__all__` is preserved. The
  new GUI, `/gui` route, and `url` form field are purely additive (SemVer minor).

## [0.5.2] - 2026-07-19

### Changed

- **Maturity classifier `3 - Alpha` → `4 - Beta`.** Reflects the actual state:
  104 functional tests at 50 % coverage, CI on Python 3.10–3.13, a corrected +
  regression-guarded router, and downstream use.
- **Pinned `pywhispercpp>=1.5`** — that line ships prebuilt wheels for macOS
  arm64, Linux x86_64/aarch64 (manylinux + musllinux) and Windows across
  cp39–cp314, so a plain `pip install vocal-helper` needs **no C/C++ toolchain**
  on any common platform (only exotic arches fall back to an sdist compile).
- **The hosted model bundle now carries the `sherpa` ONNX** (pyannote-3.0
  segmentation + TitaNet-large/small embedding), so the torch-free backend
  (`pip install vocal-helper[sherpa]`) runs offline diarization with **no
  PyTorch and no HuggingFace** — the lightest offline path. The bundle also
  dropped an unused Text-to-Speech (TTS) payload, so it stays lean.

### Documentation

- Added a **Versioning & stability** policy (README + LISEZMOI): pre-1.0, defaults
  change only in minors, never patches. Recorded honestly that the 0.5.1
  `--diar-backend` default change (`nemo` → `auto`) was part of *fixing the
  non-functional router*, not a gratuitous change — but such changes are
  minor-only from here.
- Clarified the install docs: the base install is light and wheel-based; the
  heavy `[nemo]` extra (~5 GB torch) and the model bundle are opt-in.

## [0.5.1] - 2026-07-19

### Fixed

- **The diarization router (the *aiguilleur*) is now actually enforced — it was
  decorative in 0.5.0.** The router shipped and printed a rationale but was never
  consulted with real inputs: the CLI called `select_diarization(duration_s=None)`
  (a constant → always pyannote on the auto branch), `--offline` bypassed it and
  used the `--diar-backend` default, the mic/url online path never called it, and
  `POST /pipeline` hardcoded `pyannote`. So the quality×speed **length crossover**
  that is the repo's headline value proposition could not fire in any production
  path. Now every surface routes on real conditions: files probe their duration
  (`vocal_helper.sources.probe_duration_s`, a cheap ffprobe read) so offline
  short (≤300 s, ≤4 spk) → `nemo` and long/unknown → `pyannote`; the API routes on
  the decoded buffer's real length; mic/url resolve the online backend (→ `nemo`).

### Changed

- **`--diar-backend` default `nemo` → `auto`** on both CLIs. `auto` delegates to
  the router (reporting Diarization Error Rate (DER) **and** Real-Time Factor (RTF)); an explicit `pyannote` / `nemo` /
  `sherpa` is honoured as an operator override. `POST /pipeline` `diar_backend`
  default `pyannote` → `auto` likewise.
- **`router.select_diarization` gained `nemo_available`** so a short file never
  routes to NeMo when the extra is absent — it falls through to pyannote rather
  than naming an unrunnable backend.

## [0.5.0] - 2026-07-19

### Added

- **Portable `sherpa` diarization backend (torch-free ONNX).** Selectable on both
  `OnlineDiarStage` (via `_SherpaEmbedder`) and `OfflineDiarStage` (via
  `_SherpaOfflineDiar`, sherpa-onnx `OfflineSpeakerDiarization`, whole-buffer). Runs the
  same TitaNet-large as the `nemo` backend through onnxruntime — no PyTorch, embeddable on
  every platform. Models resolve from `$VH_SHERPA_SEGMENTATION` / `$VH_SHERPA_EMBEDDING` or
  the diarization-engines bundle's `sherpa/` dir (community-1 segmentation preferred, our
  sovereign HF-free ONNX export). New `pip install vocal-helper[sherpa]` extra and
  `tests/test_diar_sherpa.py`. Study-selected in the `pasdebonneoudemauvaisesituation`
  sandbox (ADR 0002): DER 0.174 on AMI ES2011a, 0.148 on held-out IS1008a, FR+EN validated;
  streaming = periodic offline re-diarization.
- **`sherpa` selectable from the CLI.** Both front-ends now offer
  `--diar-backend sherpa` (argparse and click) — previously the backend
  existed and was tested but was unreachable from the command line
  (`choices` listed only `pyannote`/`nemo`).
- **`transcribe_pcm_with_language` helper** (`vocal_helper.asr`) returns
  `(text, detected_language)` so callers can report the language whisper
  actually used instead of echoing the request.
- **Backend router — the *aiguilleur* (`vocal_helper.router`).**
  `select_diarization(live, duration_s, max_speakers, torch_free,
  pyannote_available)` returns a `BackendPlan(mode, backend, expected_der,
  expected_rtf, reason)` that turns the measured quality×speed trade-off into one
  explicit, tested decision — both CLIs delegate to it via `_choose_file_diar`,
  so no front-end hard-codes a backend. Quality (DER) and speed (RTF) are
  first-class fields keyed from a single `_PROFILE` table, re-validated
  on-machine (`studies/router_profile_validation.py`, `pyannote.metrics` collar
  0.25) against bagarre + AMI. Findings: offline has a length crossover (nemo
  short ≤ 4 spk ↔ pyannote long/unknown/> 4 spk); online has none, so streaming
  always routes to nemo (torch-free → sherpa). `router.py` is 100 % covered
  (`tests/test_router.py`).

### Changed

- **Surfaces reshaped to a pure toolbox.** vocal-helper now exposes only
  library + argparse CLI + click CLI + FastAPI HTTP API + Model Context Protocol (MCP). The API and MCP
  surfaces are **kept**; the `POST /transcribe` response now carries the detected
  language, and `api.py` reads `__version__` dynamically (no more drift).

- **Language identification is now discovery-first — no default, no pairing.**
  `vocal_helper.lid` detection (`detect_language`, `language_posterior_curve`,
  `detect_language_regions{,_fast}`, `detect_language_speechbrain`,
  `cross_check_regions`) defaults to `supported=None`, meaning *return the
  language the input actually is* (whisper's true argmax / VoxLingua107's true
  label). A restricted candidate set is now an **opt-in routing hint**, never a
  default: `DEFAULT_SUPPORTED_LANGS` is demoted to a convenience constant with
  no privileged language. Empty or too-short-to-identify audio returns **no
  region** rather than inventing English. Completes the direction set by the
  earlier `lang_pair` removal ("language is discovered, not paired").

### Fixed

- **`POST /transcribe` reported the detected language, not the request echo.**
  With `language=auto` the endpoint now returns the language whisper detected
  from the audio instead of literally echoing `"auto"` back.
- **FastAPI app version was pinned at `0.3.7`** (four releases stale) — now
  tracks the package version (`0.4.7`).
- **`__email__` typo** (`warithmetics@deraison.ai` → `warith@deraison.ai`) and
  stale `__version__` / `CITATION.cff` (0.4.6 → 0.4.7) brought in line with
  `pyproject.toml`.

### Removed

- **Docker (`Dockerfile`) and the `GUI.md` product plan.** vocal-helper is a
  toolbox (library + CLIs + API + MCP), not a shipped container or visual
  product. Containerization stays available as self-authored deployment recipes
  in `TECHNICAL_STACK.md`; no Dockerfile is shipped.

## [0.4.7] - 2026-07-18

### Documentation

- Complete type annotations and Numpy-style docstrings for CLI/api/pipeline
  closures and the language-id helper per CODING.md.

## [0.4.6] - 2026-07-16

### Fixed

- **Offline NeMo Sortformer backend returned no speakers at all.**
  `OfflineDiarStage(backend="nemo")` parsed the model output as legacy Rich Transcription Time Marked (RTTM)
  (`SPEAKER …` lines, ≥8 fields), but nemo-toolkit 2.x emits the compact
  `"<start> <end> <speaker>"` form — so every line was dropped and the backend
  produced an empty diarization (DER 1.0 on every input). The parser now
  handles both formats (extracted to `_parse_sortformer_segments`, unit-tested
  in `tests/test_sortformer_parse.py`). The offline nemo path now works.

### Documentation

- **Offline backend crossover, measured (DER, collar 0.25).** With Sortformer
  fixed, a head-to-head on ground truth shows a length-dependent crossover:

  | corpus | offline pyannote | offline nemo (Sortformer) |
  |---|---|---|
  | AMI (20–40 min meetings) | **0.122** | 0.242 |
  | bagarre (~30 s, ≤4 spk, 26% overlap) | 0.338 | **0.177** |

  NeMo Sortformer (`diar_sortformer_4spk-v1`, end-to-end + overlap-aware, but
  4-speaker / ~90 s capped) nearly *halves* the DER on short ≤4-speaker clips,
  while pyannote 3.1 (whole-buffer, no speaker cap) wins on long meetings. So
  **pyannote stays the offline default** (robust across length + speaker count,
  best on the long inputs that dominate batch `file` use) and `--offline
  --diar-backend nemo` is the pick for short ≤4-speaker workloads. Documented
  in the `vocal_helper.diar` module docstring; `studies/diar_der_paths.py`
  reproduces the full sweep.

## [0.4.5] - 2026-07-16

### Changed

- **Batch file diarization now defaults to the reliable offline path.** A
  real-DER sweep against ground truth (`studies/diar_der_paths.py`,
  pyannote.metrics, collar 0.25) settled which diarizer to trust:

  | corpus | offline pyannote | online (no refine) | online + refine |
  |---|---|---|---|
  | AMI (real meetings) | **0.122** | 0.497 | 0.351 |
  | bagarre (26% overlap) | **0.338** | 0.586 | 0.592 |

  Offline pyannote is literature-grade (Bredin 2023 ≈ 0.188 uncollared) and
  ~3× lower DER than the online streaming diarizer, which cannot model
  overlapped speech. So `vocal-helper file --no-real-time` now **auto-selects
  the offline pyannote diarizer when its bundle is present**, falling back to
  the online diarizer + `refine_on_close` repair pass when it is not (with a
  one-line stderr note either way). New `--online` flag forces the streaming
  diarizer; explicit `--offline` is unchanged and still honours
  `--diar-backend`. Live `mic` / `url` are untouched. Downstream integrators
  embedding diarization in a larger pipeline should use `OfflineDiarStage` /
  `OfflinePipeline` for batch audio — see the `vocal_helper.diar` docstring.
  The DER sweep also independently confirms the v0.4.4 `refine_on_close` fix:
  it roughly halves the online DER on meetings that over-segment (ES2011a
  0.588 → 0.296) and never hurts. Both CLIs (argparse + click) share one
  `_choose_file_diar` policy so the default cannot drift; covered by
  `tests/test_cli_diar_default.py`.

## [0.4.4] - 2026-07-16

### Fixed

- **Online diarizer over-segmentation on long batch files.** The greedy
  single-pass online clusterer (`OnlineDiarStage`) mints a permanent new
  speaker whenever an embedding is farther than `join_threshold` from every
  existing centroid, with no cap and no merge — so on long multi-speaker
  audio, outlier embeddings (overlap, laughter, jingle, backchannels, slow
  centroid drift) each spawn a throwaway singleton, producing hundreds of
  spurious speaker labels for a handful of real speakers. `OnlineDiarStage`
  now supports a `refine_on_close` batch pass that, once the stream ends,
  globally re-clusters the collected per-segment embeddings — merging
  near-duplicate centroids (`merge_threshold`) and pruning micro-clusters
  smaller than `min_cluster_size` into their nearest surviving speaker —
  then emits the batch with corrected, compact `S<n>` labels. `vocal-helper
  file --no-real-time` (both the argparse and click CLIs) enables this
  automatically; live streaming is unchanged. An optional online
  `max_speakers` cap is also available. Covered by new model-free
  regression tests (`tests/test_diar_refine.py`).

## [0.4.3] - 2026-07-15

### Documentation

- Harmonize README/LISEZMOI to the AI Helpers common structure (single H1,
  Documentation block, PyPI + source install paths, refreshed pins to v0.4.3);
  no code changes.

## [0.4.2] - 2026-07-14

### Added

- `lid.detect_language_regions_fast` — a single-pass fast path for spoken-language
  segmentation. It runs one cheap whole-file `detect_language`; when that clears a
  confidence gate (`DEFAULT_FAST_CONF_GATE`, 0.5) on a routable language the file is
  treated as monolingual and returned as a single region, otherwise it falls back to
  the accurate posterior-curve `detect_language_regions` scan. On monolingual
  recordings this cuts per-file language identification from ~73 s to ~1 s with
  identical region output, while low-confidence (code-switched / noisy) files still
  get the full scan. Exported from the package root; covered by pure, model-free tests.

### Documentation

- Finalize suite wording: describe capabilities in plain language
  (Voice Activity Detection, Speech to Text, Speech Synthesis, source
  separation) instead of specific tool names, for consistency across the
  suite's descriptions and the documentation site.


## [0.4.1] - 2026-07-14

### Maintenance

- Apply the project coding standards across the package and `tests/`:
  Numpy-style docstrings on every function/class (including private and
  nested helpers), full type annotations with `from __future__ import
  annotations`, and comment density raised above the floor in every
  module. No public API or behavior changes.
- Route library logging through the os-helper logging surface
  (`osh.info/warning/error`) and adopt os-helper path/file utilities
  more widely; pin `os-helper>=1.5.0`.
- Refresh the project logo asset.


## [0.4.0] - 2026-07-14

### Changed

- **Offline diarization now runs whole-buffer by default.** The pdbms offline
  map-reduce study (2026-07-14 — full stack VAD + Automatic Speech Recognition (ASR) + diar on AMI, scored by
  VAD F1 / Word Error Rate (WER) / DER-JER against AMI ground truth) found DER strictly monotone
  in chunk size: whole-buffer is best (median DER **0.143** vs 0.170 at 300 s,
  and cliffs to 0.31 / 0.50 at 120 s / 60 s as speaker fragmentation outruns
  the stitch); VAD is chunk-invariant; and ASR *destabilises* when chunked (one
  meeting hit WER 1.17 on a long-window whisper loop). Accordingly
  `IDEAL_DURATION_S_PYANNOTE` is raised **300 → 3600 s**: any realistic input
  (≤ 1 h) now diarizes as a single whole-buffer call, and chunk-and-stitch
  survives only as a memory backstop past ~1 h. **NeMo is unchanged at 60 s**
  (its Sortformer 90 s cap forces chunking). Override per call with
  `OfflinePipelineConfig(diar={"ideal_duration_s": …})`.

- **No HuggingFace needed at runtime.** All model weights that used to be
  pulled from the HF hub now load from a self-hosted, HF-free bundle
  (`diarization-engines.zip`) — resolved via `resolve_diarization_engines()`
  from `$VH_DIARIZATION_ENGINES` (a local dir or URL) or the built-in default,
  cached locally and verified against a manifest. Covers the offline pyannote
  3.1 pipeline (local `config.yaml` + `.bin` weights), NeMo Sortformer (local
  `.nemo` via `restore_from` — also fixes the wrong `diar_sortformer_v1` id →
  `diar_sortformer_4spk-v1`), the online `pyannote/embedding` embedder, and the
  SpeechBrain VoxLingua107 language-ID cross-check. TitaNet already loads from
  NGC (no HF). Set `$HF_HUB_OFFLINE=1` and the full stack runs with no token,
  no HF network access. HF remains only as an automatic fallback when no bundle
  is configured.

### Added

- **Offline quality regression test (DeepEval).** `tests/test_offline_regression.py`
  runs the offline pyannote + whisper.cpp stack on a small hosted AMI subset
  (CC BY 4.0) and asserts DER / WER thresholds via custom DeepEval
  `BaseMetric`s. Marked `integration` (skipped unless models are present).

### Removed

- **`vocal_helper.tts` (PiperTTS) dropped.** Text-to-speech is out of scope for
  a diarization + transcription library and is covered far better by the
  dedicated `speaker-helper` / `speak` projects. Removes the `PiperTTS` export,
  the `[tts]` extra (`piper-tts`), and the `rhasspy/piper-voices` HF download.
  **Breaking:** import `PiperTTS` from those projects instead.

## [0.3.7] - 2026-07-13

### Changed

- **`vocal_helper.lid` is now a posterior-curve language diarizer.**
  `detect_language_regions` no longer classifies fixed windows; it samples a
  language-posterior curve over overlapping windows
  (`language_posterior_curve`), Gaussian-smooths it, takes the per-frame
  argmax, absorbs short regions, locally refines each change point (fine
  re-scan + interpolated crossing) and snaps it to the nearest silence — a
  language switch has no acoustic boundary, so it is classified and smoothed,
  not detected (after Wang et al., Interspeech 2019). Boundaries resolve to
  ~±0.4 s on a clean code-switch vs the previous ±window/2. Defaults changed:
  `DEFAULT_WINDOW_S` 20→10, `DEFAULT_MIN_REGION_S` 10→8.

### Added

- **Independent verification.** `cross_check_regions` /
  `detect_language_speechbrain` corroborate each region with a second, fully
  independent audio classifier — SpeechBrain VoxLingua107 ECAPA-TDNN — that
  shares nothing with whisper but the signal. Optional `[lid]` extra
  (`speechbrain`, `torchaudio`), imported lazily. New public exports:
  `language_posterior_curve`, `detect_language_speechbrain`,
  `cross_check_regions`, `RegionVerdict`.

## [0.3.6] - 2026-07-12

### Added

- **Acoustic language identification (`vocal_helper.lid`).** New module for
  language diarization off the same cached whisper.cpp model
  `WhisperStage` uses: `detect_language(pcm)` returns the most probable
  ISO-639-1 language among a configurable `supported` set (restricting the
  candidates keeps whisper from ranking a close relative — Galician/Catalan
  over Spanish — on short windows), and `detect_language_regions(pcm, sr)`
  slides that detection over the audio, coalesces same-language neighbours
  and absorbs regions shorter than `min_region_s` into their longer
  neighbour, returning `LangRegion(lang, t0, t1)`. Enables per-region
  transcription of code-switching recordings, where a naive whisper `"auto"`
  pass would lock onto one language and mistranscribe the rest. Exported
  from the package root.

### Changed

- Re-pinned sibling helpers to their newest tags: os-helper `v1.4.2`,
  audio-helper `v1.5.5`, capture-helper `v0.2.2`, podcast-helper `v0.3.3`.

## [0.3.5] - 2026-07-09

### Added

- **Offline full-throttle ASR batching.** `WhisperStage` gained
  `batch` + `max_chunk_s` keyword args. When `batch=True` it packs
  consecutive diarized segments into ≤ `max_chunk_s` (default 24 s)
  windows and runs **one whisper call per window** instead of one per
  segment, re-mapping each phrase back to its segment by local time
  window. whisper.cpp pads every call to a 30 s mel, so fewer/fuller
  calls amortise the fixed encoder cost. The 2026-07-09 sweep
  (`studies/asr_offline_batching.py`, 12 bilingual bagarre-rich mixes)
  measured **6.5× lower RTF (0.054 vs 0.353) at *better* WER (0.565 vs
  0.612)** — the longer decoder context cuts short-segment hallucination.
  `OfflinePipeline` enables it by default; opt back into the per-segment
  path with `OfflinePipelineConfig(asr={"batch": False})`.
- **Streaming warm-up.** `WhisperStage(warmup=True)` runs one throwaway
  inference on silence before consuming the queue, moving whisper's
  cold-start stall off the first caption. `Pipeline` (streaming) enables
  it by default.

### Notes

- Both features are additive and default-off at the `WhisperStage`
  level; the public API surface (exports, `Utterance`/`DiarizedSegment`
  keys, `S0`/`S1`/`S?` speaker labels, stage signatures) is unchanged and
  now covered by `tests/test_public_api_stability.py`. Offline still emits
  exactly one `Utterance` per diarized segment.

## [0.3.4] - 2026-07-08

### Documentation

- Cross-platform Install prerequisites (macOS / Ubuntu / Windows).

## [0.3.3] - 2026-07-07

### Fixed

- CI now installs ffmpeg (required by the ffmpeg-based load_audio) so the
  from_wav_file tests run. No package change vs 0.3.2.

## [0.3.2] - 2026-07-07

### Changed

- Decode audio via `audio_helper.load_audio` (ffmpeg) everywhere — API,
  both CLIs and `from_wav_file` now accept ANY format (mp3, m4a/AAC,
  opus, video audio tracks) and auto-resample; `from_wav_file` no longer
  requires a 16 kHz WAV. Diar/TTS writes go through `scipy.io.wavfile`.
- Remove `soundfile` entirely (code + extras); add `scipy` as a direct dep.
- Bump pins: audio-helper v1.5.4, podcast-helper v0.3.2. Bump 0.3.1 -> 0.3.2.

## [0.3.1] - 2026-07-07

## [0.3.0] — 2026-07-06

### Added

- `OnlineDiarStage` and `OfflineDiarStage` now accept a `device`
  kwarg (`"cpu"` / `"cuda"` / `"mps"` / `None`). Default `None`
  uses the new `_auto_torch_device` helper which picks CUDA > MPS >
  Central Processing Unit (CPU). On Apple Silicon this lifts pyannote 3.1 from ~ 15× real-time
  (CPU) to roughly real-time (MPS). `_PyannoteOfflineDiar.load`
  wraps `pipeline.to(...)` in a `try` block ; if MPS rejects an op
  the stage stays on CPU rather than crash. `_PyannoteEmbedder`
  forwards the device to `pyannote.audio.Inference(..., device=)`
  the same way. The previously-skipped
  `test_offline_pipeline_vs_youtube_captions` integration test is
  re-enabled.
- `vocal_helper._settings` — hand-rolled `settings.yaml` loader (no
  PyYAML dep) and `resolve_hf_token()` helper. The CLI, the example
  script, and `OnlineDiarStage` / `OfflineDiarStage` now resolve the
  HuggingFace token in this order: explicit value > `HF_TOKEN` env
  var > `secrets.hf_token` in `settings.yaml`.
- `settings.yaml.example` — git-tracked template; copy to
  `settings.yaml` (git-ignored) and fill in the real token. The
  placeholder `hf_XXXX` is treated as unset.
- Override the lookup with `$VOCAL_HELPER_SETTINGS` to pin a specific
  file (handy for tests and unusual deploys).
- Test suite expanded from 3 → 40 cases — split into
  `tests/test_smoke.py`, `test_settings.py`, `test_sources.py`,
  `test_pipeline.py`, `test_cli.py`. A `conftest.py` autouse fixture
  isolates `$HF_TOKEN` / `$VOCAL_HELPER_SETTINGS` so the developer's
  real `settings.yaml` cannot leak into CI assertions.
- `.github/workflows/ci.yml` hardened with `lint`
  (ruff check + ruff format --check informational), `test` (Python
  matrix 3.10/3.11/3.12/3.13 + coverage report on 3.12), and
  `pre-commit` (mirror of the local commit-time hooks). Adds pip
  caching and per-ref concurrency cancellation.
- Whisper **bias prompt** — `WhisperStage(initial_prompt=…)` and
  `transcribe_pcm(initial_prompt=…)`, surfaced as `--initial-prompt`
  on every CLI. Empty by default; a domain-aligned prompt cut WER
  15–25 pp and saved up to 39 % RTF on the 2026-06-30 AMI sweep.
- **`SemanticEOTStage`** (`vocal_helper.eot`) — opt-in, LiveKit-style
  turn detector that holds back VAD segments that look mid-thought and
  merges them with their successor. Enable with `PipelineConfig(eot=…)`
  or `--eot` / `--eot-model`; off by default (one extra LLM hop per
  voiced segment). Offline path deliberately has no EOT block.
- **Multi-surface exposure** — the same pipeline is now reachable via
  the argparse CLI (`vocal-helper`), a click twin (`vocal-helper-click`),
  a FastAPI HTTP app (`vocal_helper.api`), and an MCP server
  (`vocal-helper-mcp`), all sharing one config builder.
- `vocal_helper.tts` (local Piper TTS) and a
  `vocal_helper.parallel_pipelines` demonstrator.

### Changed

- Pre-existing ruff lint errors in `vocal_helper/diar.py`,
  `vocal_helper/pipeline.py`, `vocal_helper/sources.py` and
  `vocal_helper/vad.py` cleaned up (unused `step` / `frame_period_s`
  vars, missing `strict=` on `zip`, six try/except/pass blocks
  rewritten as `contextlib.suppress`, import-order). `ruff check .`
  now passes from a clean checkout.
- Ruff exclusion list moved from `tests/` to `studies/` — one-off
  research scripts are no longer linted, but tests now are.
- **Default online-diar backend** switched `pyannote` → `nemo`
  (TitaNet) per the 2026-06-30 embedding sweep (+76 % separability
  margin on AMI). Pass `backend="pyannote"` to skip the ~5 GB NeMo
  install.
- **Default LLM analyst model** switched `gemma4:e4b` → `gemma3:4b`
  per the 2026-06-30 7-model Pareto sweep (3× faster *and* higher
  cos_sim). Reflected across the library and every CLI surface.
- `vocal_helper.cli` is now a thin backward-compat shim; the canonical
  argparse implementation lives in `vocal_helper.cli_argparse`, so the
  four surfaces share a single config builder with no drift.
- Removed the attribution-audit CI job and pre-commit hook;
  `nomoreclaude.sh` is no longer tracked (kept as a local-only tool).

## [0.1.0] — 2026-06-30

Initial release.

### Added

- `vocal_helper.types` — `PcmFrame`, `VoicedSegment`, `DiarizedSegment`, `Utterance`, `SummarySnapshot` typed dicts.
- `vocal_helper.sources` — `from_microphone` (capture-helper), `from_wav_file`, `from_numpy_array`.
- `vocal_helper.vad.SileroVADStage` — Silero v5 ONNX VAD with run-by-run emission and edge-padded segments.
- `vocal_helper.diar.OnlineDiarStage` — online cosine-distance running-mean clustering, `pyannote` and `nemo` backends, `join_threshold=0.30` default.
- `vocal_helper.asr.WhisperStage` — pywhispercpp turbo wrapper with thread-pool execution and word timestamps.
- `vocal_helper.llm.GemmaAnalystStage` — Ollama-served `gemma4:e4b` rolling-summary analyst, recent window 60 s, summarises every 5 evicted utterances.
- `vocal_helper.pipeline.Pipeline` — top-level orchestrator with `subscribe_voiced` / `subscribe_diarized` / `subscribe_utterances` fan-out hooks.
- `vocal-helper` CLI with `mic` and `file` subcommands.
- `examples/live_mic_to_text.py` — end-to-end live demo.
- Smoke tests in `tests/test_smoke.py`.
