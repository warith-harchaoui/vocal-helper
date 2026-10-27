# Examples

[🇫🇷 EXEMPLES.md](https://github.com/warith-harchaoui/vocal-helper/blob/main/EXEMPLES.md) · 🇬🇧 English

Hand-picked recipes for the shapes people actually need: a live microphone, a
recorded file, a relay into your own code, and the measurement that tells you
whether any of it worked.

Every snippet here was checked against the source. The ones that need nothing
but CPU were **run**, and their output below is what they printed — pasted, not
described. The ones that need model weights, a microphone or a served language
model are marked, and they were verified against the code rather than executed.

## Table of Contents

- [Setup](#setup)
- [1. Live mic → terminal transcript](#1-live-mic--terminal-transcript)
- [2. Replay a WAV at burst speed (offline batch)](#2-replay-a-wav-at-burst-speed-offline-batch)
- [3. A rolling summary over a meeting](#3-a-rolling-summary-over-a-meeting)
- [4. Custom WebSocket relay via the subscriber API](#4-custom-websocket-relay-via-the-subscriber-api)
- [5. Diarize without PyTorch, and tune the clustering](#5-diarize-without-pytorch-and-tune-the-clustering)
- [6. One-shot transcription, no pipeline](#6-one-shot-transcription-no-pipeline)
- [7. Score a transcript: why cpWER and not WER](#7-score-a-transcript-why-cpwer-and-not-wer)
- [8. Ask the router which backend to use](#8-ask-the-router-which-backend-to-use)
- [Command line](#command-line)
- [Troubleshooting](#troubleshooting)

## Setup

```bash
pip install 'vocal-helper[all]'
cp settings.yaml.example settings.yaml   # carries the diarization-engines URL
```

The LLM analyst (example 3, and the `--llm` flag) additionally needs a serving
backend up. `best-engine-ai-helper` picks the model for your machine from the
committed `vocal_helper/llm.brief.yaml`; the resolved engine file names the
exact `ollama pull` to run:

```bash
ollama serve
```

No HuggingFace token is needed anywhere. Every model weight comes from the
self-hosted diarization-engines bundle named in `settings.yaml`
(`engines.diarization_url`) — see
[the README](README.md#model-weights-no-huggingface-needed).

Every snippet below assumes these three imports, and nothing else:

```python
import asyncio

import numpy as np

import vocal_helper as voh
```

`voh` is the suite's canonical alias for this package — the same one the
docstrings and the README use, so code copied between them keeps working. A
snippet that needs more than those three names imports it inline, right where
it is used.

## 1. Live mic → terminal transcript

Needs the `[mic]` extra and a microphone, so this one was not executed.

```bash
vocal-helper mic \
  --initial-prompt "Réunion d'équipe : design, marketing, planning, livrables"
```

`--initial-prompt` is **strongly recommended**, not decoration: the 2026-06-30
sweep on AMI dev-slice (`studies/whisper_prompt_lang_lock.py`) measured a
domain-aligned bias prompt dropping WER by 15-25 percentage points and saving up
to 39 % RTF. Name your conversational domain and a handful of proper nouns or
technical terms you expect to hear.

The same thing in Python (the shipped demo is `examples/live_mic_to_text.py`):

```python
import asyncio

import vocal_helper as voh


async def main():
    pipeline = voh.Pipeline(
        source=lambda: voh.sources.from_microphone(),
        config=voh.PipelineConfig(
            asr={
                "language": "auto",  # discovered from the audio, no default
                "initial_prompt": "Réunion d'équipe : design, marketing, planning, livrables",
            },
        ),
    )
    async for ev in pipeline.run():
        if "text" in ev:
            print(f"[{ev['t0']:.1f}s {ev['speaker']}] {ev['text']}")


asyncio.run(main())
```

## 2. Replay a WAV at burst speed (offline batch)

```bash
vocal-helper file ./conversation.wav --no-real-time --jsonl > out.jsonl
```

`--no-real-time` drops the wall-clock pacing, so frames fire as fast as they
decode, and it also flips the run onto the **offline** whole-buffer diarizer
when one is installed — the reliable path for batch. `--jsonl` emits one JSON
event per line, which is what you want piping into `jq` or a store.

The router writes its decision to stderr before the first utterance, so a
redirected stdout stays clean JSON:

```
vocal-helper: router → offline pyannote (quality DER 0.122, speed RTF 0.067; …). Pass --diar-backend to override.
```

## 3. A rolling summary over a meeting

Needs a served language model, so this one was not executed.

```python
import asyncio

import vocal_helper as voh


async def main():
    pipeline = voh.OfflinePipeline(
        source=lambda: voh.sources.from_wav_file("./meeting.wav", real_time=False),
        config=voh.OfflinePipelineConfig(
            asr={"language": "auto"},  # discovered from the audio, no default
            llm={
                "engine": voh.resolve_engine(),  # best-engine-ai-helper picks the model
                "recent_window_s": 60.0,  # the last 60 s stay verbatim
                "flush_every_s": 60.0,  # refresh per 60 s of evicted speech
            },
        ),
    )
    async for ev in pipeline.run():
        if "summary" in ev:
            print("\n--- rolling summary ---")
            print(ev["summary"])


asyncio.run(main())
```

The `llm` dict is splatted straight into `GemmaAnalystStage`, so its keys are
that constructor's parameter names. There is **no `model` key**: since `2.0.0`
the model is never named here — `voh.resolve_engine()` returns the descriptor
`best-engine-ai-helper` resolved for this machine, and passing a model tag
raises `TypeError`. `flush_every_s` takes precedence over the count-based
`flush_every_n`, which applies only when `flush_every_s=None`.

## 4. Custom WebSocket relay via the subscriber API

`subscribe_utterances` observes the stage without consuming the merged event
stream, so the `async for` loop below stays free to do something else. Needs
`aiohttp` and a microphone, so this one was not executed.

```python
import asyncio
import json

from aiohttp import web

import vocal_helper as voh

clients: set[web.WebSocketResponse] = set()


async def on_utterance(u):
    payload = json.dumps(
        {
            "t0": u["t0"],
            "t1": u["t1"],
            "speaker": u["speaker"],
            "text": u["text"],
        }
    )
    for ws in list(clients):
        try:
            await ws.send_str(payload)
        except Exception:
            clients.discard(ws)


async def ws_handler(req):
    ws = web.WebSocketResponse()
    await ws.prepare(req)
    clients.add(ws)
    async for _ in ws:
        pass
    clients.discard(ws)
    return ws


async def run_pipeline():
    pipeline = voh.Pipeline(source=lambda: voh.sources.from_microphone())
    pipeline.subscribe_utterances(on_utterance)
    async for _ in pipeline.run():
        pass


async def main():
    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8765)
    await site.start()
    await run_pipeline()


asyncio.run(main())
```

`subscribe_voiced` and `subscribe_diarized` tap the two earlier stages the same
way, which is how a caption reaches the screen before the transcript exists.

## 5. Diarize without PyTorch, and tune the clustering

The `sherpa` backend is the same TitaNet-large through onnxruntime: no torch, no
CUDA, embeddable on iOS and Android. Needs the sherpa models from the bundle, so
this one was not executed.

```python
import asyncio

import vocal_helper as voh


async def main():
    pipeline = voh.OfflinePipeline(
        source=lambda: voh.sources.from_wav_file("./call.wav", real_time=False),
        config=voh.OfflinePipelineConfig(
            diar={
                "backend": "sherpa",
                "sherpa_num_clusters": 2,  # a known 2-party call: skip the auto-detector
            },
            asr={"language": "auto"},
        ),
    )
    async for ev in pipeline.run():
        if "text" in ev:
            print(f"[{ev['t0']:7.2f}s {ev['speaker']}] {ev['text']}")


asyncio.run(main())
```

`sherpa_num_clusters=2` is not a detail. The auto-detector's threshold was tuned
on clean meeting audio and over-segments noisy telephony badly — the measured
case left roughly thirty speakers on a two-party call. When you know the count,
say it.

On the streaming side the equivalent lever is `join_threshold`, the cosine
distance below which a segment joins an existing speaker:

```python
config = voh.PipelineConfig(
    diar={"backend": "nemo", "join_threshold": 0.60},  # tighter than the 0.50 default
    asr={"language": "auto"},
)
```

`nemo` is already the default online embedder, so what this changes is only the
threshold. Raising it merges more readily, which is what you want when one
person keeps being split into several; lowering it splits more readily, for when
two people are being fused into one. The shipped **0.50** is itself the result of
raising it from 0.30: on 8 AMI meetings with 4 speakers each, 0.30 invented 87
speakers and 0.50 found 28, and 0.60 found a count closer to the truth while
forcing genuinely different people together — the unrecoverable direction.

## 6. One-shot transcription, no pipeline

**Executed.** One PCM buffer in, text out: no VAD, no diarization, no summary.
The language is discovered from the audio by default (`language="auto"`) — there
is no default language and no language pairing.

```python
import numpy as np

import vocal_helper as voh

pcm = np.zeros(16_000 * 5, dtype=np.float32)  # five seconds of digital silence
text = voh.transcribe_pcm(pcm, sr=16_000)  # language auto-discovered
print(repr(text))
```

```
'Thank you.'
```

That output is not a typo, and it is the most useful thing this file can show
you: given silence, whisper invents a short plausible phrase rather than
returning nothing. Do not feed it un-gated audio and trust the result — which is
exactly what the VAD stage in front of the pipeline is for.

Need the language whisper actually settled on? The sibling helper returns it
beside the text:

```python
from vocal_helper.asr import transcribe_pcm_with_language

text, language = transcribe_pcm_with_language(pcm, sr=16_000)
print(repr(language), repr(text))
```

```
None 'Thank you.'
```

`None` is the honest answer here: there was no speech to identify. On real
speech this is an ISO-639-1 code such as `'fr'`.

## 7. Score a transcript: why cpWER and not WER

**Executed.** Two speakers, every single word transcribed correctly, and the
diarizer merged them into one. Plain WER cannot see the mistake; cpWER prices
the words and the speaker together.

```python
import vocal_helper as voh

reference = [
    {"speaker": "A", "text": "on se voit mardi pour la revue de design"},
    {"speaker": "B", "text": "parfait je prepare les maquettes"},
]
# Same words, but the diarizer collapsed both voices into one speaker.
merged = [
    {"speaker": "S0", "text": "on se voit mardi pour la revue de design"},
    {"speaker": "S0", "text": "parfait je prepare les maquettes"},
]

flat = lambda turns: " ".join(t["text"] for t in turns)
print(voh.bench.wer_report(flat(reference), flat(merged))["wer"])
print(voh.bench.cpwer(reference, merged))
```

```
0.0
{'cpwer': 0.7143, 'errors': 10, 'ref_words': 14, 'n_ref_speakers': 2, 'n_hyp_speakers': 1, 'mapping': {'S0': 'A'}, 'style_tolerant': False}
```

A perfect 0.0 next to a 0.71: that gap is the whole argument for the module.
`n_ref_speakers` against `n_hyp_speakers` says at a glance that the damage is
attribution and not transcription, so you know which stage to go and fix.

The companion metric answers "**which** words were missed", which a rate never
does:

```python
print(voh.bench.term_error_rate("le Figaro publie demain", "le Fegaro publie demain", ["Figaro"]))
```

```
{'present': 1, 'found': 0, 'missed': ['Figaro'], 'error_rate': 1.0}
```

One letter wrong is a rounding error in WER and a lost record in any
corpus-level search.

## 8. Ask the router which backend to use

**Executed.** Pure Python, no model loaded: the *aiguilleur* answers from the
measured profile, and it reports quality and speed rather than just a name.

```python
import vocal_helper as voh

plan = voh.select_diarization(live=False, duration_s=45.0, max_speakers=3)
print(plan.backend, plan.expected_der, plan.expected_rtf)
print(plan.reason)
print(voh.select_diarization(live=False, duration_s=1800.0).backend)
print(voh.select_diarization(live=True, duration_s=None).backend)
print(voh.select_diarization(live=False, duration_s=600.0, torch_free=True).backend)
```

```
nemo 0.142 0.051
≤300s, ≤4 speakers → nemo Sortformer (end-to-end, confusion ~0; DER 0.142, RTF 0.051)
pyannote
nemo
sherpa
```

Short and dense goes to `nemo`, long-form to `pyannote`, a live stream to
`nemo`, and a machine with no PyTorch to `sherpa`. The CLI and the HTTP surface
call this same function, so the decision never has to be reproduced by hand —
and reading `plan.reason` is how you check it rather than trust it.

## Command line

One entry point, four subcommands. **Executed** — this is the real `--help`:

```
$ vocal-helper --help
Usage: vocal-helper [OPTIONS] COMMAND [ARGS]...

  Turn audio into diarized, transcribed utterances — live or from a file.

  Each subcommand feeds one source (microphone, WAV, URL) through the VAD →
  diarization → ASR pipeline, with an optional rolling LLM summary. The
  "transcribe" command is the one-shot escape hatch: no VAD, no diarization.

Options:
  -V, --version  Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  file        Replay a 16 kHz mono WAV through the pipeline.
  mic         Live microphone input (needs the [mic] extra).
  transcribe  One-shot transcription of a WAV file (skip VAD / diarization).
  url         Stream from any URL yt-dlp can reach (needs the [stream]...
```

```bash
vocal-helper -V                                   # vocal-helper, version 3.3.0

vocal-helper transcribe clip.wav                  # text only, language discovered
vocal-helper transcribe clip.wav --jsonl          # {"path": …, "text": …} on stdout

vocal-helper file meeting.wav --offline --llm     # whole-buffer diarization + summary
vocal-helper file meeting.wav --no-real-time --jsonl > out.jsonl

vocal-helper mic --llm --eot                      # live, with the semantic turn detector
vocal-helper url "https://youtu.be/…" --language fr

vocal-helper file meeting.wav --diar-backend sherpa   # override the router
```

`--whisper-model`, `--language`, `--threads`, `--initial-prompt`,
`--diar-backend`, `--join-threshold`, `--llm`, `--llm-recent-window-s`,
`--endpoint`, `--eot` and `--jsonl` are shared by `mic`, `file` and `url`;
`vocal-helper <subcommand> --help` prints the full surface. `--language`
defaults to `auto`, so pass it only to *force* a language rather than discover
one.

## Troubleshooting

**`vocal-helper: no offline diarizer installed — using the online diarizer with
the refine pass.`** The batch path wanted the whole-buffer diarizer and could
not find one. Install `vocal-helper[pyannote]` and copy `settings.yaml.example`
to `settings.yaml` so the bundle can be fetched. The run still completes, at a
worse DER.

**`StageCrashed`** out of `run()`. Since `3.0.0` a pipeline that lost a stage
raises instead of returning an empty transcript with the reason buried in a log.
Read the exceptions off the pipeline:

```python
from vocal_helper.pipeline import StageCrashed

try:
    async for ev in pipeline.run():
        ...
except StageCrashed:
    for stage, err in pipeline.stage_failures:
        print(stage, type(err).__name__, err)
```

`voh.Pipeline(source=…, strict=False)` restores the old leniency;
`stage_failures` is filled either way, so even the lenient path can be asked
what went wrong.

**A transcript appears where there was only silence.** See example 6: whisper
hallucinates on digital silence. Keep the VAD stage in front of ASR, and treat
`transcribe_pcm` on un-gated audio as a smoke test rather than a measurement.

**Dozens of speakers on a two-party phone call.** The `sherpa` auto-detector was
tuned on clean meeting audio. Pass `sherpa_num_clusters=2`, or raise
`sherpa_cluster_threshold` above its `0.5` default.

**The LLM analyst hangs or errors on the first utterance.** `resolve_engine()`
only resolves the descriptor; it does not start the server. Run `ollama serve`
(or point `--endpoint` at a remote host) and check that the generated
`vocal_helper/llm.engine.yaml` names a model you have pulled.

**`ModuleNotFoundError` on a source.** `from_microphone` needs the `[mic]`
extra, `from_url` needs `[stream]`, and the HTTP / MCP surfaces need
`[api,mcp]`. The base install deliberately carries only the file and numpy
sources.
