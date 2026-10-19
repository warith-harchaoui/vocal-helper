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
| Project | Live streaming | Online diarization | Offline diarization | Local only | Rolling LLM summary | Multi-source input | Ergonomic Python API | Multi-surface |
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
2D representation of the table above, generated with [standpoint](https://github.com/warith-harchaoui/standpoint).

![Positioning map](https://raw.githubusercontent.com/warith-harchaoui/vocal-helper/main/assets/landscape.png)

The axes read **horizontal: Adaptable ↔ Versatile**, **vertical: Independent ↔ Comprehensive**,
and they carry **72 %** of the information in the eight criteria (46 % and 26 %). So read the map
as a shape, not a scoreboard — two points close together agree on most of what was scored, and
may still differ on the criterion you personally care about.

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

## When to pick something else

The honest half of a landscape document.

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
