# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

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
- `.github/workflows/ci.yml` hardened with three jobs: `lint`
  (ruff check + ruff format --check informational), `no-more-claude`
  (runs the audit script below), and `test` (Python matrix
  3.10/3.11/3.12/3.13 + coverage report on 3.12). Adds pip caching
  and per-ref concurrency cancellation.
- `nomoreclaude.sh` — portable audit script that scans every commit
  and every tracked file for Claude/Anthropic mentions. Default mode
  is dry-run (exits 1 on findings); `--apply` rewrites history with
  `git-filter-repo` (or `git filter-branch` fallback); `--apply
  --push` force-pushes after confirmation. Wired into CI as a hard
  gate.

### Changed

- Pre-existing ruff lint errors in `vocal_helper/diar.py`,
  `vocal_helper/pipeline.py`, `vocal_helper/sources.py` and
  `vocal_helper/vad.py` cleaned up (unused `step` / `frame_period_s`
  vars, missing `strict=` on `zip`, six try/except/pass blocks
  rewritten as `contextlib.suppress`, import-order). `ruff check .`
  now passes from a clean checkout.
- Ruff exclusion list moved from `tests/` to `studies/` — one-off
  research scripts are no longer linted, but tests now are.

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
