# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org).

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
