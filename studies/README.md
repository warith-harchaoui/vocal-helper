# Studies

Reproducible sweeps that motivate every non-obvious default in
`vocal-helper`. If you read a "default X, calibrated on the 2026-06-30
sweep" comment in the source, the sweep script lives here.

Every study follows the same shape :

1. **Corpus / hypothesis** — what conversation dataset was used and
   what was being measured.
2. **Sweep** — a `main()` that iterates the knob(s) and writes a JSON
   log to `run_logs/`.
3. **Verdict** — a numbered `figures/` panel and a bullet in the
   module docstring's `Verdict :` block.

To reproduce, install the base + relevant extras and run the script.
Every sweep pins its random seeds, so identical input tapes must
produce identical numbers within numerical noise.

## Index

| Study | What it decided | Verdict |
|---|---|---|
| [`diar_embedding_backend.py`](diar_embedding_backend.py) | Which per-segment speaker embedder to default to for `OnlineDiarStage` | **TitaNet (NeMo) beats pyannote/embedding by 76 % separability margin** on AMI dev-slice (2 meetings, N ≥ 8). Default `backend="nemo"`. |
| [`diar_falcon_vs_pyannote.py`](diar_falcon_vs_pyannote.py) | Whether NVIDIA Sortformer beats pyannote 3.1 as the offline backend | pyannote 3.1 stays the offline default ; Sortformer is opt-in via `backend="nemo"`. |
| [`llm_cadence_sweep.py`](llm_cadence_sweep.py) | How often the analyst should refresh the rolling summary | Single-meeting sweep on AMI IS1008a : `flush_every_s=60` is the RTF × cosine-similarity Pareto optimum. |
| [`llm_cadence_sweep_multi.py`](llm_cadence_sweep_multi.py) | Same cadence question, cross-checked over ≥ 4 meetings | Multi-meeting confirms `flush_every_s=60` is robust to per-conversation noise. |
| [`llm_engine_comparison.py`](llm_engine_comparison.py) | Ollama vs llama.cpp vs vLLM for the analyst hop | Ollama picked for the operator-friendly install path. |
| [`llm_model_size_sweep.py`](llm_model_size_sweep.py) | Which Ollama model to default to | `gemma4:e4b` : Pareto optimum on RTF × summary quality on Apple Silicon. |
| [`whisper_prompt_lang_lock.py`](whisper_prompt_lang_lock.py) | Whether an `initial_prompt` + `language=` combo reduces WER | Prompt bias cuts WER **15-25 pp** and RTF **up to 39 %** on AMI when the domain vocabulary is known. `--initial-prompt` exposed on the CLI. |
| [`stt_faster_whisper_vs_pywhispercpp.py`](stt_faster_whisper_vs_pywhispercpp.py) | Whether to switch the ASR backend on CUDA | Reserved for `v0.2` — CTranslate2 wins on NVIDIA GPUs but the CPU / Apple Silicon path stays on whisper.cpp. |
| [`eot_semantic_vs_silero.py`](eot_semantic_vs_silero.py) | Whether a semantic end-of-turn gate lifts the false-cutoff rate | LiveKit-style semantic EOT ships as opt-in `SemanticEOTStage` behind the `[llm]` extra. |

## Reproducing

Base environment :

```bash
pip install -e '.[pyannote,llm,dev]'
```

Corpus-specific extras :

- **AMI meeting corpus** — download from
  <https://groups.inf.ed.ac.uk/ami/download/> ; point studies at the
  root via `AMI_ROOT=/path/to/ami-corpus`.
- **HF-gated pyannote 3.1** — set `HF_TOKEN` or write it into
  `settings.yaml` as documented in the README.
- **Ollama** — the LLM cadence sweeps require Gemma pulled locally :
  `ollama pull gemma4:e4b`.

Each script writes its raw numbers to `run_logs/` in
newline-delimited JSON, then a matching PNG or SVG under
`figures/` via `doc/figures/_gen_figures.py`. The `run_logs/` and
`figures/` folders are checked in — reviewers should be able to
plot without rerunning if they only want to inspect.

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
