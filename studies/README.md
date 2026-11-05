# Studies

Reproducible sweeps that motivate every non-obvious default in
`vocal-helper`. If you read a "default X, calibrated on the 2026-06-30
sweep" comment in the source, the sweep script lives here.

Every study follows the same shape :

1. **Corpus / hypothesis** — what conversation dataset was used and
   what was being measured.
2. **Sweep** — a `main()` that iterates the knob(s) and writes its raw
   numbers next to the script (`<study>.json`, sometimes a `<study>.log`).
3. **Verdict** — a bullet or a table in the module docstring's
   `Verdict :` block, the part a reader can check without rerunning.

To reproduce, install the base + relevant extras and run the script.
Every sweep pins its random seeds, so identical input tapes must
produce identical numbers within numerical noise.

## Index

| Study | What it decided | Verdict |
|---|---|---|
| [`diar_embedding_backend.py`](diar_embedding_backend.py) | Which per-segment speaker embedder to default to for `OnlineDiarStage` | **TitaNet (NeMo) beats pyannote/embedding by 76 % separability margin** on AMI dev-slice (2 meetings, N ≥ 8). `backend="auto"` takes `nemo` first when the `[nemo]` extra is installed. |
| [`diar_falcon_vs_pyannote.py`](diar_falcon_vs_pyannote.py) | Whether NVIDIA Sortformer beats pyannote 3.1 as the offline backend | pyannote 3.1 stays the offline default ; Sortformer is opt-in via `backend="nemo"`. Since superseded: community-1 in `3.0.0`, Nemotron 3 ahead of it in `4.0.0`. |
| [`llm_cadence_sweep.py`](llm_cadence_sweep.py) | How often the analyst should refresh the rolling summary | Single-meeting sweep on AMI IS1008a : `flush_every_s=60` is the RTF × cosine-similarity Pareto optimum. |
| [`llm_cadence_sweep_multi.py`](llm_cadence_sweep_multi.py) | Same cadence question, cross-checked over ≥ 4 meetings | Multi-meeting confirms `flush_every_s=60` is robust to per-conversation noise. |
| [`llm_engine_comparison.py`](llm_engine_comparison.py) | Ollama vs llama.cpp vs vLLM for the analyst hop | Ollama picked for the operator-friendly install path. |
| [`llm_model_size_sweep.py`](llm_model_size_sweep.py) | Which Ollama model to default to | `gemma4:e4b` : Pareto optimum on RTF × summary quality on Apple Silicon. No tag is pinned any more: best-engine-ai-helper resolves the model from `vocal_helper/llm.brief.yaml`, and this sweep informs its `min_tps` / `headroom`. |
| [`whisper_prompt_lang_lock.py`](whisper_prompt_lang_lock.py) | Whether an `initial_prompt` + `language=` combo reduces WER | Prompt bias cuts WER **15-25 pp** and RTF **up to 39 %** on AMI when the domain vocabulary is known. `--initial-prompt` exposed on the CLI. |
| [`stt_faster_whisper_vs_pywhispercpp.py`](stt_faster_whisper_vs_pywhispercpp.py) | Whether to switch the ASR backend on CUDA | Reserved for `v0.2` — CTranslate2 wins on NVIDIA GPUs but the CPU / Apple Silicon path stays on whisper.cpp. |
| [`eot_semantic_vs_silero.py`](eot_semantic_vs_silero.py) | Whether a semantic end-of-turn gate lifts the false-cutoff rate | LiveKit-style semantic EOT ships as opt-in `SemanticEOTStage` behind the `[llm]` extra. |
| [`asr_nemotron_online.py`](asr_nemotron_online.py) | How to stream Nemotron 3.5 with sliding windows | **2 s hop, 16 s left context, 0.48 s lookahead**: WER within 0-0.05 of offline at 1.7 s mean delay; lookahead is the lever that matters (none costs 0.05-0.11), context beyond 16 s buys nothing. |
| [`asr_live_whisper_vs_nemotron.py`](asr_live_whisper_vs_nemotron.py) | Which ASR engine to run live | **Nemotron per segment** beats whisper per segment on all four corpora: WER 0.24 / 0.38 / 0.27 / 0.48 against 0.25 / 0.53 / 0.79 / 0.83 (AMI, overlapped mixes, CV-fr, French TV), RTF ≈ 0.1. Sliding-window streaming keeps 2.6 s of delay on continuous speech but costs RTF 0.4-1.1. |
| [`asr_nemotron_vs_whisper_offline.py`](asr_nemotron_vs_whisper_offline.py) | Which ASR engine to run offline, and whether discovering the language helps whisper | **Nemotron 3.5 (ONNX int8) by default when the language is to be discovered**: WER 0.23 / 0.38 / 0.27 on AMI / overlapped mixes / CV-fr against 0.38 / 0.54 / 0.69 for whisper `auto`, RTF int8 0.02-0.07 on CPU; whisper keeps clean English (0.012) and, given the language, French TV. Language discovery before whisper: 0.69 → 0.28 on CV-fr. |
| [`vad_nemotron_vs_silero.py`](vad_nemotron_vs_silero.py) | Whether Nemotron's diarizer activity or CTC blank can replace Silero as the speech detector | **Silero stays**: AUC 0.964 on AMI against 0.929 (diarizer) and 0.923 (smoothed CTC); the diarizer ties Silero on overlapped mixes; raw CTC is not a speech detector (AUC 0.60-0.73). |
| [`asr_offline_batching.py`](asr_offline_batching.py) | Whether the offline whisper path should concatenate segments | **≤ 24 s chunks**: RTF 6.5× lower than per-segment calls, at a better WER (median 0.565 against 0.612). |
| [`asr_acceleration_rtf.py`](asr_acceleration_rtf.py) | Which local ASR runtime is fastest here, and whether a live placement holds | whisper.cpp on Metal at RTF 0.033, ahead of MLX (0.061) and CPU (0.164); stage RTFs add per compute unit, not across them. Write-up: [`asr_acceleration_rtf.md`](asr_acceleration_rtf.md). |
| [`router_profile_validation.py`](router_profile_validation.py) | The DER and RTF behind each row of `vocal_helper.router` | Re-validated on bagarre (30 mixes) and AMI dev-slice, collar 0.25, median: the numbers in `_PROFILE`. |
| [`diar_growing_buffer.py`](diar_growing_buffer.py) | Whether re-diarizing the whole buffer periodically brings offline quality live | **0.176 median DER** on the eight AMI dev-slice meetings, the full-file answer, at RTF 0.10-0.16: `GrowingBufferDiarStage`, the live fallback. |
| [`diar_nemotron_streaming.py`](diar_nemotron_streaming.py) | Nemotron 3 Diarization against our chain, under our protocol | Optimal mapping, collar 0.25, on the same AMI slice; the numbers that led to `NemotronOnnxDiar`. |
| [`diar_far_field.py`](diar_far_field.py) | Whether Nemotron or pyannote holds up at a distance, on a mix with exact truth | **Nemotron 0.090 / 0.105 / 0.252** against pyannote 0.299 / 0.296 / 0.553 (anechoic, RT60 0.3 s, 0.6 s), right speaker count everywhere; its missed speech stays flat, so AMI's annotation explains its AMI score. |
| [`rediar_live_der.py`](rediar_live_der.py) | What the bounded periodic re-diarization (`rediar`) is really worth | On five minutes of AMI IS1008a it halved the online error (0.666 → 0.348, offline 0.149). On the whole IS1008a and ES2011a meetings it did worse than the online stage (0.679 / 0.817 against 0.597 / 0.616). `growing` replaced it as the default. |

## Reproducing

Base environment :

```bash
pip install -e '.[pyannote,llm,dev]'
```

Corpus-specific extras :

- **AMI meeting corpus** — download from
  <https://groups.inf.ed.ac.uk/ami/download/> ; point the studies at
  the dev-slice with `VH_AMI_ROOT` (see below).
- **Model weights** — the diarization-engines bundle named in
  `settings.yaml`, as documented in the README. Two older sweeps
  (`diar_embedding_backend.py`, `diar_falcon_vs_pyannote.py`) still read
  `HF_TOKEN` to pull from the hub.
- **Ollama** — the LLM cadence sweeps require Gemma pulled locally :
  `ollama pull gemma4:e4b`.

### Where the data lives

No study names a machine: every location is an environment variable,
read by [`_paths.py`](_paths.py). When a variable is unset, the default
is a folder of this checkout (`data/`, `models/`, `run-logs/`, all
git-ignored), so a fresh clone runs the same command once the data is
dropped in place.

| Variable | Points at | Default |
|---|---|---|
| `VH_STUDY_DATA_ROOT` | The bench data root: `ami/dev-slice/`, `bagarre-rich/`, `common-voice-subset/`, `speech.wav` | `data/` |
| `VH_AMI_ROOT` | The AMI dev-slice, one folder per meeting with `mix.wav` + `words.rttm` | `$VH_STUDY_DATA_ROOT/ami/dev-slice` |
| `VH_TV_CORPUS` | The French TV episodes built by the paid transcription path | `$VH_STUDY_DATA_ROOT/francetv-corpus/output` |
| `VH_RUN_LOGS` | Where a long sweep tees its log (and `doc/figures/_gen_figures.py` reads it back) | `run-logs/` |
| `VH_MODELS_DIR` | Local whisper weights, ggml and MLX (`asr_acceleration_rtf.py`) | `models/` |
| `VH_BENCH_SRC` | The diarization bench's `src/`, imported by `diar_embedding_backend.py` | none: the script stops and names it |
| `VH_BENCH_ROOT` | The bench checkout passed as `--bench` in the usage lines | none (an argument) |
| `VH_NEMOTRON_STUDY` | The Nemotron export folder holding `lite/` and `onnx/` | none (an argument) |
| `VH_PAID_WALLET` | The ledger caching the paid diarizations (`diar_vs_paid_reference.py --wallet`) | none: required |
| `HF_TOKEN` | The Hugging Face token for the gated pulls | none |

Each script writes its raw numbers next to itself (`<study>.json`,
sometimes a `<study>.log`). Those outputs are git-ignored; four of
the older sweeps also have a figure under `doc/figures/`, drawn by
`doc/figures/_gen_figures.py`. The verdict a reader can check without
rerunning is the one copied into the script's docstring.

## Naming convention

`<stage>_<what-varies>.py` :

- `diar_*` — anything under `vocal_helper.diar`.
- `llm_*` — Gemma analyst.
- `whisper_*` — ASR knobs.
- `eot_*` — end-of-turn gating.
- `stt_*` — cross-engine STT comparisons.

New sweeps land in a matching PR that also updates :

1. The module docstring that consumes the new default (with the
   `Verdict :` block).
2. `CHANGELOG.md` under `Changed`.
3. This table.
