# Landscape

[🇫🇷 PAYSAGE.md](https://github.com/warith-harchaoui/vocal-helper/blob/main/PAYSAGE.md) · 🇬🇧 English

Related and competing projects in the "live speech → speaker-labelled text → summary" space,
scored against the job `vocal-helper` is built for: an **async producer/consumer pipeline
turning a live PCM stream (mic, URL or file) into diarized, transcribed utterances, plus an
optional rolling LLM summary**. A project built for another job — batch transcription,
offline-only diarization — is not penalised for it; the score is fit to *this* niche, nothing
more.

Updated 2026-10-19. What changed since the August table is not only the entries: several of
these ratings now come from **our own measurements** rather than from impressions, and the
measurements moved two of them.

## What we measured ourselves

| finding | where it lands in the table |
|---|---|
| pyannote **community-1** ties 3.1 at 0.116 median DER on AMI, but its worst meeting is **0.149 against 0.474** — same centre, a tail four times shorter | `pyannote.audio` scores 5 offline |
| On three real 64-minute six-speaker panels, over-splitting was **2.7× (community-1), 3.0× (3.1), 3.7× (Sortformer), 19.7× (sherpa-onnx)** | why `sherpa-onnx` scores 2 offline despite being an excellent portable runtime |
| **Sortformer 4spk-v1 never emits a fifth speaker** — an architectural ceiling, confirmed on 30 five-speaker mixes | `NVIDIA NeMo` capped at 4 offline |
| **diart** reached 6.04 % speaker error against 11.7 % for the community-1 path on a long two-speaker call, with a first label in **500 ms against 20 s** | `diart` scores 5 online |

That last one deserves a caveat the number alone hides: the call had **two** speakers, which is
diart's easy case, and the literature consistently reports it as *less* accurate than
clustering-based pipelines on harder multi-speaker material. Read it as "fastest to a usable
label", not "most accurate".

## At a glance

<!-- TABLE:START -->
| Live Transcription | Live streaming | Online diarization | Offline diarization | Local only | Rolling LLM summary | Multi-source input | Ergonomic Python API | Multi-surface |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **vocal-helper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| whisper.cpp | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| faster-whisper | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| WhisperX | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| whisper_streaming | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| pyannote.audio | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| diart | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| NVIDIA NeMo | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐ |
| Nemotron-3 Diarization | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ |
| sherpa-onnx | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| FunASR | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Moonshine | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| SpeechBrain | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐ |
| vosk | ⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Deepgram | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Gladia | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| pyannoteAI | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
<!-- TABLE:END -->

## Positioning map

<!-- FIGURE:START -->
2D representation of the table above.

![Positioning map](https://raw.githubusercontent.com/warith-harchaoui/vocal-helper/main/assets/landscape.png)

The map is a 2-D summary of the eight criteria, so read it as a shape, not a scoreboard. `vocal-helper` is at the top-right corner. The axes read **Horizontal: Adaptable ↔ Versatile** and **Vertical: Independent ↔ Comprehensive**.

The grouping is the interesting part. The cloud services (Deepgram, Gladia) sit near
`vocal-helper` on *capability* and far from it on independence, which is the trade they offer.
The pure speech-to-text engines cluster bottom-right: independent, versatile, and with no
opinion about who is speaking. The diarization research stacks (pyannote.audio, NeMo,
Nemotron-3, SpeechBrain) sit left: deep on one axis, not built to be a pipeline.
<!-- FIGURE:END -->

## Positioning

`vocal-helper` sits where **whisper.cpp's ergonomics** (local, cheap, no GPU strictly needed)
meet the **live diarization and rolling analyst** that most stacks push to a batch layer. It
does not try to beat pyannote at diarization or whisper at transcription — it uses them, and
owns the part nobody else owns: the producer/consumer chain between them, and the measurement of
what that chain costs the person watching it.

Concretely, what is here and rare elsewhere:

- **One package for both regimes, with a measured default in each.** community-1 offline, diart
  online. Two regimes, two winners, neither assumed.
- **Scoring that matches the failure** (`vocal_helper.bench`): cpWER prices words and speaker
  together, because a diarizer that merged five panellists into two scores exactly like one that
  separated them under plain WER.
- **Live measurement** (`vocal_helper.live_metrics`): latency from end-of-speech, revision rate,
  per-queue wait. Offline has WER and DER; live had nothing, including in the projects above.
- **Failures that are loud.** A lost stage raises instead of returning an empty transcript.

## When to pick what

A comparison that never recommends a competitor is advertising, so this section
goes both ways.

**Pick `vocal-helper`** when you want speaker-labelled text *while the audio is
still arriving*, on your own machine, and you would rather configure a pipeline
than assemble one — and when the same code has to handle a recorded file later,
with the diarization backend chosen for you from the file's real length instead
of guessed.

**Pick something else** when:

- **You only need a transcript of a file.** Use `faster-whisper` or `whisper.cpp` directly.
  Everything here about queues and latency is overhead you will not use.
- **You need the best offline diarization and nothing else.** Use `pyannote.audio` with
  community-1 — which is exactly what `vocal-helper` does under the hood, so the only question
  is whether you want the pipeline around it.
- **You need Mandarin, Cantonese, Japanese or Korean.** `FunASR` is built for them and this is
  not.
- **You need to run on a Raspberry Pi.** `Moonshine` is designed for that hardware; our local
  path targets a laptop-class machine.
- **You want an SLA, not a library.** Deepgram, Gladia, AssemblyAI. The trade is sovereignty:
  audio leaves the machine, and the bill scales with hours.

## Sources and as-of date

**As of 2026-10-19.** Every rating above is a claim about software that keeps
moving, so the date is part of the claim.

The table mixes two kinds of evidence, and they are not worth the same.

**Measured here.** The findings under *What we measured ourselves*, and every
DER / RTF figure this repository quotes, come from a script in this repository
run against a named corpus — never from a vendor's own number:

| Claim | Where it was run | Corpus |
|---|---|---|
| per-scenario DER + RTF behind the router | [`studies/router_profile_validation.py`](studies/router_profile_validation.py) | 30 short mixes (`bagarre`) + AMI dev-slice, `pyannote.metrics` collar 0.25, median |
| the online embedder default (`nemo`) | [`studies/diar_embedding_backend.py`](studies/diar_embedding_backend.py) | AMI dev-slice, inter- minus intra-speaker cosine margin |
| community-1 against 3.1, median and tail | [`CHANGELOG.md`](CHANGELOG.md) under `3.0.0` | AMI, 8 meetings, collar 0, overlap scored |
| over-splitting on real panels, including sherpa-onnx's 19.7× | [`DIARIZATION-TROUBLES.md`](DIARIZATION-TROUBLES.md) §9-§10 | three 64-minute six-speaker panels |
| what an `initial_prompt` is worth in WER and RTF | [`studies/whisper_prompt_lang_lock.py`](studies/whisper_prompt_lang_lock.py) | AMI dev-slice |
| the summary cadence (`flush_every_s=60`) | [`studies/llm_cadence_sweep_multi.py`](studies/llm_cadence_sweep_multi.py) | ≥ 4 AMI meetings |

Two caveats a reader is owed. The audio for the router validation and the
`sherpa` row (ADR 0002) lives in a **private** companion repository
(`pasdebonneoudemauvaisesituation`), so those runs are reproducible by us and
not by you. The streaming-diarizer ranking and the `diart` comparison come from
two further non-public repositories — `pdbms` (sweep of 2026-06-29, N = 2089 per
system) and `operator-assistance-streaming`. They are cited for provenance, and
they are weaker evidence than the rows you can open.

**Read, not measured.** Every other cell is our reading of that project's own
repository and documentation on the as-of date: what it ships and claims to
support. Where a capability was a matter of degree we rated down rather than up,
and where a project is built for another job the number means "fit to *this*
niche", never "worse software".

The upstream work this package actually builds on is kept as BibTeX in
[`doc/refs.bib`](doc/refs.bib) — among others
[whisper.cpp](https://github.com/ggerganov/whisper.cpp),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[NVIDIA NeMo](https://github.com/NVIDIA/NeMo),
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx),
[pyannote community-1](https://www.pyannote.ai/blog/community-1) and Bredin's
INTERSPEECH 2023 paper for the published DER band the pyannote pipeline is read
against.

No benchmark figure appears above for Deepgram, Gladia, AssemblyAI or
pyannoteAI. We have not run them against this corpus, and printing a vendor's
own number beside ours would look like a comparison when it is not one.

## Ideas worth stealing

Researched rather than guessed: what each neighbour does better, and what this library would have to change to match it. Nothing here is a commitment — it is the reading list the next version starts from.

- **`WhisperX` — forced alignment for word timestamps that are actually word-level** — It runs wav2vec2 phoneme alignment after transcription, reaching sub-100 ms word timestamps, and batches VAD-segmented audio at roughly 70× realtime on one GPU. Whisper's own timestamps are segment-level guesses by comparison.

- **`WhisperLiveKit` — an API surface clients already speak** — It exposes OpenAI- and Deepgram-compatible endpoints, so existing code points at it with a base-URL change. A local-first tool that is drop-in for the hosted API it replaces removes the main reason people stay on the hosted one.

- **Streaming Sortformer and Simul-Whisper — the current low-latency pair** — Streaming diarization that commits speaker turns incrementally, rather than re-clustering the whole window each time, is what keeps latency flat as a conversation runs long.

- **`Moonshine` and sherpa-onnx Zipformer — the CPU-only tier** — Both are built for low latency without a GPU. A live transcriber that degrades to a smaller streaming model instead of failing is more useful than one that requires CUDA, and the suite already routes model choice through best-engine-ai-helper.
