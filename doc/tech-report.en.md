---
title: "vocal-helper — Technical Report"
subtitle: "Producer/consumer voice pipeline : empirical defaults, design notes, deployment levers"
author: Warith HARCHAOUI
date: Summer 2026
bibliography: refs.bib
---

# Abstract

`vocal-helper` is an asynchronous **producer / consumer voice
pipeline** for `Python ≥ 3.10`. It turns a live or recorded PCM
stream into diarized, transcribed utterances and (optionally) a
rolling LLM summary of the conversation. Two pipelines ship :
`Pipeline` for streaming use cases (microphone, URL, podcast feed),
and `OfflinePipeline` for batch use cases (meeting recordings,
voicemail archives). Every default in this report is justified by
an empirical study under `studies/` whose log + JSON output lives
on the user's research drive. The library is BSD-3-Clause licensed,
local-first, and built to play nicely with the rest of the
[AI Helpers](https://harchaoui.org/warith/ai-helpers) suite (the
upstream research codebase is `pdbms` [@pdbms]).

# 1. Goals & non-goals

## 1.1 Goals

- **Live mic → diarized transcript + rolling LLM summary**, all
  local, on Apple Silicon or commodity Linux + NVIDIA, with no
  cloud round-trip in the default path.
- **Offline batch** on a meeting recording — highest-quality
  diarization (`pyannote/speaker-diarization-3.1`
  [@pyannotediarization], auto-chunk + stitch past 300 s) → ASR
  → optional summary.
- **Empirically justified defaults.** Every knob is backed by a
  reproducible study on the AMI Meeting Corpus
  [@carletta2007unleashing] dev-slice. No marketing-blog defaults.
- **Producer / consumer queues** between stages so each stage runs
  at its own cadence — the LLM analyst stays at RTF ≈ 0.1 while
  the VAD runs at RTF ≈ 1e-5.

## 1.2 Non-goals

- No voiceprint enrolment. Speaker IDs stay anonymous `S0`, `S1`, …
  within a session ; never persisted to a named identity across
  sessions. Industrial-deployment compliance constraint.
- No token / character-streaming LLM output. Every LLM call returns
  the full response when ready. UX preference.
- No WebRTC / multi-participant transport in v0.x. Use
  `livekit-agents` [@livekitagents] for that.
- No TTS for v0.1.0 (added optionally in v0.2.0 via
  `vocal_helper.tts.PiperTTS` — see §4).

# 2. Pipeline architecture

## 2.1 Online pipeline

```
[Source]   →  [VAD]   →  [Online Diar]  →  [STT]   →  [LLM analyst (optional)]
  PCM         voiced     speaker-tagged     text          rolling summary
  frames      segments   segments
```

Optionally, a `SemanticEOTStage` sits between `VAD` and `Online
Diar` (opt-in, §3.5).

## 2.2 Offline pipeline

```
[Source]   →  [Offline Diar]  →  [STT]  →  [LLM analyst (optional)]
  full        full-buffer        text       rolling summary
  PCM         pyannote 3.1
              + chunk+stitch
              past 300 s
```

## 2.3 Edges

Every arrow is an `asyncio.Queue` bounded at `qsize_pcm = 200`
(4 s of audio in flight at 20 ms / frame) or `qsize_seg = 32`. The
`None` sentinel propagates cleanly through every stage on
shutdown. Subscribers (`subscribe_voiced` / `subscribe_diarized`
/ `subscribe_utterances`) fan out to side consumers without
back-pressuring the main chain.

# 3. Per-stage design & default choices

## 3.1 VAD — Silero v5

Default `SileroVADStage(activity_threshold=0.5, min_silence_ms=300,
min_speech_ms=300, edge_pad_ms=200, sample_rate=16000)` using
Silero v5 [@silero] ONNX on CPU. The 48 ms cadence + 0.5 threshold
operating point is the canonical pdbms setting validated in the
upstream `vad-cadence-study.md` §10. The 300 ms silence threshold
sits in the conversational sweet spot reported by both LiveKit
[@livekitturnblog] and the foundational Sacks–Schegloff–Jefferson
turn-taking framework [@sacks1974simplest] — humans naturally
respond after 200–300 ms gaps.

## 3.2 Online diarization — TitaNet embeddings (default)

`OnlineDiarStage` consumes `VoicedSegment` events from VAD,
embeds each one once, and runs a per-segment cosine running-mean
clusterer over the global speaker list. Three embedding backends are
wired :

- `backend='nemo'` — NVIDIA TitaNet [@titanet] via NeMo [@nemo].
- `backend='pyannote'` — `pyannote/embedding` [@pyannoteembedding].
- `backend='sherpa'` — the same TitaNet-large run through
  `sherpa-onnx`/onnxruntime, **torch-free** : it installs light and
  embeds on any platform (DER 0.174, FR+EN validated; ADR 0002). Its
  ONNX weights ship in the diarization-engines bundle, so the path runs
  with no PyTorch and no HuggingFace.

**Default is `nemo` (TitaNet)**, selected by the Summer 2026
embedding-backend sweep
(`studies/diar_embedding_backend.py`) on AMI N=2 :

| backend  | median intra-cosine | median inter-cosine | **margin (inter − intra)** | wall / call |
|----------|---------------------:|---------------------:|---------------------------:|------------:|
| pyannote | 0.739 | 0.940 | 0.201 | **6 ms** |
| **TitaNet** | 0.560 | 0.915 | **0.354** | 45 ms |

![Diar embedding backend — separability margin and per-call wall time on AMI](figures/fig-diar-embedding-backend.svg)

TitaNet's separability margin is **76 % wider**, at 7× per-call
latency — negligible per voiced segment in a streaming workload.
The cost is install footprint : NeMo + torch is ~ 5 GB ; pass
`backend='pyannote'` to opt out.

`join_threshold = 0.30` and `ema_alpha = 0.1` were inherited from
the pdbms Summer 2026 stitch-threshold sweep on `ChunkedOfflineDiarizer`
(the cosine plateau on AMI dev-slice N=8 sits at {0.30, 0.35, 0.40}
with median DER 0.135 vs baseline 0.116). The same calibration
transfers because the embedding distribution is the same.

## 3.3 Offline diarization — pyannote 3.1 with auto chunking

`OfflineDiarStage` consumes the full PCM buffer end-to-end and
hands it to `pyannote/speaker-diarization-3.1` [@pyannotediarization].
For inputs longer than `ideal_duration_s` (300 s for pyannote, 60 s
for the NeMo Sortformer [@sortformer] alternative) the audio is
chunked with `overlap_s = 10 s` and stitched via cosine AHC at
`stitch_threshold = 0.35` (the centre of the pdbms plateau). The
NeMo Sortformer path remains opt-in : it dominates on ≤ 60 s clips
but hangs past its 90 s training cap, so vocal-helper does not
expose it as the default.

The torch-free `sherpa` backend clusters the whole buffer inside one
`sherpa-onnx` call, so `stitch_threshold` never applies to it. Its
clustering was previously hardcoded — `FastClustering` threshold `0.5`
and speaker count `-1` (auto). `0.5` was tuned on clean AMI meeting
audio; on noisy, PII-redacted 2-party telephony it over-segments into
~36 speakers. Since **v0.7.0**, `OfflineDiarStage` plumbs
`sherpa_cluster_threshold` and `sherpa_num_clusters` through to that
config (defaults unchanged). A 2026-07-23 sweep against a pyannoteAI
silver ground truth showed raising the threshold reduces this only
slowly (~30 speakers at `0.6`), whereas `sherpa_num_clusters=2`
collapses telephony to the correct count cleanly — the value to use
when the speaker count is known (2-party calls).

## 3.4 STT — pywhispercpp turbo with bias prompt

`WhisperStage(model="large-v3-turbo-q5_0", language="auto",
threads=6, word_timestamps=True, initial_prompt="",
min_segment_ms=250)` wrapping pywhispercpp [@pywhispercpp] →
whisper.cpp [@whispercpp] → OpenAI Whisper [@radford2023whisper]
turbo [@whisperturbo].

The single most impactful lever is `initial_prompt`. The Summer 2026
sweep (`studies/whisper_prompt_lang_lock.py`) on AMI :

| config | meeting | WER | RTF |
|---|---|---:|---:|
| no prompt | IS1008a | 0.505 | 0.044 |
| **+ bias prompt** | IS1008a | **0.351** | 0.043 |
| no prompt | ES2011a | 0.625 | 0.067 |
| **+ bias prompt** | ES2011a | **0.380** | 0.041 |

![Whisper bias prompt — WER drop on AMI dev-slice](figures/fig-whisper-bias-prompt.svg)

A domain-aligned bias prompt drops WER by **15-25 percentage
points** and saves up to **39 % RTF**. The prompt should name the
domain and a handful of expected proper nouns or technical terms.
Default is the empty string (zero-config path works) but the
docstring + CLI help + EXAMPLES.md all push the caller to provide
one.

Language locking (`language="en"` vs `"auto"`) has a negligible
effect on quality in this sweep — keep `"auto"` unless you have a
strong production reason to pin.

**STT engine comparison — pywhispercpp vs faster-whisper**
(`studies/stt_faster_whisper_vs_pywhispercpp.py`, Summer 2026) :

| engine | IS1008a WER | IS1008a RTF | ES2011a WER | ES2011a RTF |
|---|---:|---:|---:|---:|
| **pywhispercpp** | 0.358 | **0.037** | 1.398 ⚠ | **0.039** |
| faster-whisper [@fasterwhisper; @ctranslate2] | 0.360 | 0.342 | 0.466 | 0.337 |

`pywhispercpp` is **~ 10× faster** on Apple Silicon (Metal-native
inference) at parity WER on the clean meeting, but
**catastrophically hallucinates on ES2011a** (WER 1.398 means it
emits more text than reference). `faster-whisper` is slower
(CTranslate2's CPU path) but more robust. Default stays
`pywhispercpp` for the streaming RTF advantage ; future work
should add hallucination detection (token-perplexity threshold)
with a `faster-whisper` fallback rather than swap engines wholesale.

## 3.5 Semantic end-of-turn — opt-in

`SemanticEOTStage` (opt-in via `PipelineConfig.eot`) sits between
VAD and online diar. For every incoming `VoicedSegment` :

1. Whisper-transcribe a partial (same model the downstream
   `WhisperStage` uses, kept in a thread pool).
2. Ask a small classifier LLM (`qwen2.5:3b` [@qwen25] by default)
   whether the partial transcript is a complete thought.
3. If complete → emit. If incomplete → buffer it, wait for the
   next segment, merge, re-classify. Force-emit after
   `max_merge_s = 4 s` of accumulation.

Inspiration : LiveKit's turn-detector v1.0 [@livekitturndetector]
(2026-04), which distilled Qwen2.5-0.5B [@qwen25] from a
Qwen2.5-7B teacher, fusing a semantic and an acoustic branch.
LiveKit reports 9.9 % false-cutoff at 300 ms median semantic
latency.

**Honest result on AMI IS1008a** (`studies/eot_semantic_vs_silero.py`,
Summer 2026) :

| config | n_segments | n_false_cuts | false_cut_rate |
|---|---:|---:|---:|
| Silero VAD only (baseline) | 160 | 40 | 0.250 |
| Silero + `SemanticEOTStage` | 148 | 38 | 0.257 |

Our semantic-only first cut **does not improve false-cut rate** on
AMI compared to the silence-threshold-only baseline. Median
classifier latency is ~230 ms per call (vs LiveKit's 10-25 ms),
so the engineering trade-off currently goes the wrong way too. The
stage is shipped opt-in and kept in the codebase as a scaffold for
a future drop-in of LiveKit's actual distilled turn-detector
model — the qwen2.5:3b general-purpose classifier is too coarse
for this task at this size.

## 3.6 LLM analyst — Gemma 3 4b with time-based cadence

`GemmaAnalystStage(model="gemma3:4b", recent_window_s=60.0,
flush_every_s=60.0, flush_every_n=5)` wrapping Ollama [@ollama].

The default model `gemma3:4b` [@gemma3] is the Pareto winner of
the Summer 2026 7-model sweep
(`studies/llm_model_size_sweep.py`) on AMI IS1008a with cadence
`flush_every_s=60` :

| model | RTF | cos_sim |
|---|---:|---:|
| gemma4:e2b-mlx [@gemma4] | 0.193 | 0.456 |
| gemma4:e4b-mlx (prior default) | 0.313 | 0.420 |
| gemma4:12b-mlx | 2.453 | **0.496** |
| **gemma3:4b** | **0.099** | **0.466** |
| qwen2.5:3b [@qwen25] | 0.043 | 0.399 |
| qwen3:8b [@qwen3] | 1.628 | 0.350 |
| llama3.2:3b [@llama32] | 0.066 | 0.367 |

![LLM analyst — 7-model Pareto on AMI IS1008a (gemma3:4b is the chosen operating point)](figures/fig-llm-model-pareto.svg)

`gemma3:4b` **dominates the prior default on both axes** : 3 ×
faster RTF AND higher cos_sim vs an offline single-shot reference
summary. The Pareto front also exposes `gemma4:12b-mlx` (RTF
2.453, cos_sim 0.496) for offline batch and `qwen2.5:3b` (RTF
0.043, cos_sim 0.399) for tight RTF budgets.

The cadence default `flush_every_s = 60` was selected by two
complementary sweeps :

- single-meeting (`studies/llm_cadence_sweep.py`) on AMI IS1008a :
  t=60s wins cos_sim (0.420).
- multi-meeting median (`studies/llm_cadence_sweep_multi.py`) on
  IS1008a + ES2011a + ES2011d + TS3004a : t=60s gives RTF 0.278
  / cos_sim 0.339 ; n=20 gives RTF 0.369 / cos_sim 0.354. The
  0.015 cos_sim gap is within inter-meeting noise (cos_sim ranges
  0.279 – 0.471 for the same config), and t=60s is 25 % faster.

![LLM cadence — single-meeting vs multi-meeting Pareto (t=60s is the chosen operating point)](figures/fig-llm-cadence.svg)

## 3.7 LLM serving engine — Ollama (default), with adaptive policy

The engine comparison `studies/llm_engine_comparison.py` finds
Ollama [@ollama] is the only engine that loads cleanly on Apple
Silicon today (vLLM [@vllm] support for Metal is experimental ;
mlx-lm [@mlxlm] requires manual install). On other hardware
(Linux + NVIDIA), vLLM would be the recommended engine.

The roadmap envisages an auto-detection policy : Ollama with
MLX-tagged weights on macOS, vLLM on Linux + CUDA, llama.cpp /
Ollama with gguf as the universal fallback. The lever is
unimplemented in v0.1.0 — callers can override via
`GemmaAnalystStage.host` or by swapping the `model` tag.

## 3.8 TTS — Piper (opt-in)

`vocal_helper.tts.PiperTTS` wraps Piper [@piper] for local CPU-only
neural TTS. Default voice `en_US-amy-medium` (English) and
`fr_FR-siwis-medium` (French) ; the full catalogue is on
`rhasspy/piper-voices` [@pipervoices]. Synth is one-shot — no
chunk streaming, matching the no-character-stream constraint —
and exposed as a helper that the caller wires into their own
subscriber (typically on `Utterance` or `SummarySnapshot` events)
rather than as a pipeline stage. This keeps vocal-helper
transcription-first while making the audio-out loop closeable in
two lines of caller code.

# 4. Open directions

## 4.1 Single-language Whisper distillation

The current default `large-v3-turbo-q5_0` [@whisperturbo] is
OpenAI's October 2024 distillation of large-v3 to a 4-layer
decoder, kept multilingual. Deployments locked to one language
have three escalating cost levels :

1. **Zero training — published per-language distillations.**
   For English, `distil-whisper/distil-large-v3`
   [@distilwhisper; @distilwhispercard] reports ~ 6 × throughput
   at parity WER. For French, Bofeng Huang's family
   [@bofenghuangfrench] : `whisper-large-v3-french` (full
   fine-tune), `-distil-dec16` (~ 2 ×), `-distil-dec8` (~ 4 ×),
   `-distil-dec4` (~ 8 × — matches turbo's 4-layer decoder
   topology, French-only). Integration : either swap whisper.cpp
   for `faster-whisper` [@fasterwhisper; @ctranslate2] which loads
   HF checkpoints directly, or convert HF weights to `.gguf` via
   `whisper.cpp`'s `models/convert-h5-to-ggml.py` and quantise
   them to `q5_0` for `pywhispercpp`.

2. **Fine-tune existing turbo to one language (1-3 GPU days).**
   Continue training `large-v3-turbo` on Common Voice
   [@commonvoice] + Multilingual LibriSpeech
   [@multilinguallibrispeech] + the user's production corpus with
   the language token locked and a low learning rate (~1e-6).
   Expected : same parameter count, +2 to +5 pp WER on the target
   language, no speed gain. Cost : 20-50 A100-hours.

3. **Full Distil-Whisper-style distillation of turbo to a
   language-specialised student (1-3 weeks GPU).** Architecture :
   4-layer decoder student (same as turbo), full encoder kept ;
   loss : KL-divergence on teacher logits + cross-entropy on
   ground-truth ; corpus : 5 000-20 000 h target language
   (pseudo-labels from the multilingual teacher are acceptable).
   Cost : 500-2 000 A100-hours. Outcome : 2-4 × speedup over
   multilingual turbo at parity-or-better WER on the target
   language.

The recommended near-term path is **(1)**. The Bofeng
distillations already sit at the operating point we'd target.
**(3)** is justified only as an open-source contribution, not a
deployment shortcut.

## 4.2 LiveKit-grade EOT distillation

Our `SemanticEOTStage` uses a general-purpose Qwen2.5-3B classifier
at ~ 50-200 ms / call. LiveKit's turn-detector v1.0
[@livekitturndetector] runs a 0.5B-param student at 10-25 ms / call
with a dual-branch semantic + acoustic architecture. Catching up
requires a small-scale distillation project (their training
recipe is published in the 2026-04 blog post). Out of scope for
v0.1.0.

## 4.3 Pipecat-style typed frames + barge-in

`Pipecat 1.0` [@pipecat; @pipecatdocs] organises events into
`SystemFrame` / `DataFrame` / `ControlFrame` lanes, with system
frames bypassing the data queue for immediate processing — the
foundation of their barge-in (`InterruptionFrame`) and clean
shutdown protocols. vocal-helper's current `None` sentinel
propagation works for shutdown but does not give out-of-band
priority to interruption signals. A small refactor in v0.3 would
adopt the dual-queue lane pattern, enabling barge-in once a TTS
playback loop is wired (currently `PiperTTS` is a synth helper, not
an autonomous audio-out stage).

## 4.4 Falcon Speaker Diarization

Picovoice claims [@picovoicefalcon] their Falcon diarizer hits
DER 10.3 % vs pyannote 9.0 % on their benchmark, with **221 × less
compute and 15 × less memory**. JER even favours Falcon
(−7.5 pp). Subject to the closed-source licence + access-key
trade-off, this would be a serious offline backend candidate for
server-heavy deployments — `studies/diar_falcon_vs_pyannote.py`
will quantify the AMI-specific delta as soon as a Picovoice
access key is provisioned.

# 5. Reproducibility

Every quoted number in this report has a study script and a JSON
output :

| Study | Script | JSON output |
|---|---|---|
| Stitch threshold (pdbms upstream) | `pdbms-scratch/pyannote_stitch_threshold_sweep_2026-06-30.py` | `…/pyannote_stitch_threshold_sweep_2026-06-30.log` |
| LLM cadence single | `vocal-helper/studies/llm_cadence_sweep.py` | `…/vocal_helper_llm_cadence_2026-06-30.json` |
| LLM cadence multi | `vocal-helper/studies/llm_cadence_sweep_multi.py` | `…/vocal_helper_llm_cadence_multi_2026-06-30.json` |
| Whisper prompt × lang | `vocal-helper/studies/whisper_prompt_lang_lock.py` | `…/vocal_helper_whisper_prompt_lang_2026-06-30.json` |
| LLM engine comparison | `vocal-helper/studies/llm_engine_comparison.py` | `…/vocal_helper_llm_engine_2026-06-30.json` |
| LLM 7-model sweep | `vocal-helper/studies/llm_model_size_sweep.py` | `…/vocal_helper_llm_model_size_2026-06-30.json` |
| Diar embedding backend | `vocal-helper/studies/diar_embedding_backend.py` | `…/vocal_helper_diar_embedding_2026-06-30.json` |
| Faster-whisper vs pywhispercpp | `vocal-helper/studies/stt_faster_whisper_vs_pywhispercpp.py` | (running) |
| Falcon vs pyannote | `vocal-helper/studies/diar_falcon_vs_pyannote.py` | (pending PV key) |
| EOT semantic vs Silero | `vocal-helper/studies/eot_semantic_vs_silero.py` | (running) |

All JSON results live on the user's research drive at
`/Volumes/orange-dev/extra/pdbms-scratch/run-logs/`. Each script
is self-contained — re-running it from a clean machine reproduces
the same numbers up to model-loading noise (~ ±5 % wall time).

# 6. See also

- The full competitive landscape table is in
  [`LANDSCAPE.md`](../LANDSCAPE.md) — rows : competitors / tools,
  columns : characteristics, cells : ★1–★5.
- The upstream research codebase that feeds vocal-helper's
  defaults : `pdbms` [@pdbms], with its own
  `doc/tech-report.{en,fr}.md`.
- The AI Helpers landing page : <https://harchaoui.org/warith/ai-helpers>.

# References
