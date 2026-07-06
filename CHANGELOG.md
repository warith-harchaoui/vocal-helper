# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.3.0] — 2026-07-06

### Added

- `OnlineDiarStage` and `OfflineDiarStage` now accept a `device`
  kwarg (`"cpu"` / `"cuda"` / `"mps"` / `None`). Default `None`
  uses the new `_auto_torch_device` helper which picks CUDA > MPS >
  CPU. On Apple Silicon this lifts pyannote 3.1 from ~ 15× real-time
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
