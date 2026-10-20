# Making diarization reliable, offline and online — what the research says, and what we do

*2026-10-20. Priority chantier for vocal-helper.*

The honest starting position, from our own measurements rather than impressions:

| path | AMI | verdict |
|---|---|---|
| offline, pyannote community-1 | DER **0.116** median, 0.149 worst | good |
| offline, sherpa ONNX (what we ship HF-free) | DER **0.148–0.174** *with the speaker count given* | good, conditionally |
| offline, sherpa, count auto-detected | finds **103–124** speakers on a 6-person panel | **unusable** |
| online, greedy clusterer | **87** speakers for 4 raw; 0.074 confusion after repair | weak |

So the problem is not really "DER is too high". **It is that nothing can count speakers**, and
DER hides that: the metric is duration-weighted, so dropping a guest who speaks for ninety
seconds barely moves it. That is not our opinion, it is the finding of *Beyond DER* (arXiv
[2609.27315](https://arxiv.org/abs/2609.27315)), which proposes JER plus explicit counting
metrics for exactly this reason — and it matches the metric set already chosen in our own
notes before we read it.

## What the literature actually offers us

### 1. The cheapest win is not a model — it is telling the diarizer what you know

pyannoteAI's own engineers put it plainly: *"Pass `numSpeakers` when you have the exact
count, and estimation leaves the loop entirely"*, and capturing speaker metadata at source
(meeting invites, participant lists) is *"the cheapest accuracy improvement available in
diarization"*. Our measurements agree violently — the same sherpa path scores 0.148 with
the count and 103–124 without it.

**Done (2026-10-20).** `num_speakers` / `min_speakers` / `max_speakers` now reach both
backends through `OfflineDiarStage.diarize`. A floor is the preferred hint: a panel roster
proves those people are distinct but is not a total, since episodes also carry reporters,
archive clips and narrators nobody announced.

**Why it must arrive at the clustering and not after:** we measured that an over-estimated
count *cannot* be merged back down using the same embedding that produced it — AUC 0.60
median, 0.27 worst, at or below chance, because the clusterer already spent exactly that
information deciding those groups were far apart.

### 2. Voiceprint enrolment turns an open-set problem into a closed-set one

pyannoteAI again: enrol *"voiceprints for recurring participants to prevent drift-based
phantom speakers"*. This is the single highest-leverage idea for our actual corpus: *C dans
l'air* is 501 episodes with a recurring cast. Blind clustering per episode is solving the
hard problem 501 times; identification against enrolled voices is the easy one.

**Status:** the machinery exists in notes-helper (`voiceprints`, `people.db`). What is
missing is enrolment from the episodes already built, and a vocal-helper-side API so it is
not one consumer's private trick.

### 3. Online: the per-segment greedy clusterer is a dead end, and now we have proof

No setting of it recovers the speaker count — best case still 28 labels for 4 before repair,
7 after. Tuning bought a real improvement (see below) but not a fix.

Two architectures are worth the port:

**FASTDIAR** (Microsoft, arXiv [2610.02941](https://arxiv.org/abs/2610.02941), **MIT**,
weights on GitHub releases — *not* HuggingFace, so it clears our hard rule).
A causal frame-level encoder emitting an embedding every 80 ms from a 2 s past window, plus
online clustering gated on embedding confidence. Numbers that matter to us:

- **5× faster than real time on a single CPU thread, no GPU** — this is the "groupe
  électrogène" constraint satisfied, and it is also what makes mobile plausible;
- **no speaker cap**, degrading 1.2–1.5× beyond four speakers where Streaming Sortformer
  degrades 2.3–4.4×. Our panels are six-plus;
- models are 2.4 M / 4.6 M / 10.4 M parameters — small enough that ONNX export is routine;
- 960 ms fixed latency; 12.35 % DER on VoxConverse, 21.58 % on NOTSOFAR excluding overlap.

**Nemotron-3-Diarization** (NVIDIA, Sept 2026, 100 M params, OpenMDW, commercial use
allowed). DIHARD III 12.73 % at the 30.4 s profile and **13.28 % at 0.64 s** — nearly as
good streaming as offline, so one model could cover both of our modes. Capped at 8
speakers. Ships for NVIDIA GPUs on Linux via NeMo, so an M2 Max needs ONNX→CoreML export;
100 M parameters makes that reasonable, and we have done exactly this for
pyannote-segmentation and TitaNet already. A ModelScope mirror exists, which matters because
the HuggingFace route is closed to us.

### 4. Nobody has solved counting

Worth stating so we stop expecting a model to rescue us. EEND-EDA and EEND-TA *under*-count
in crowded recordings, silently dropping low-activity speakers. Clustering methods
over-count instead: *"too permissive gives under-counting, too strict gives over-counting"* —
which is precisely the trade-off curve we measured for our own merge threshold. An
independent 2026 test on a fourteen-speaker recording found frameworks estimating between
four and eight. The field's answer is to **constrain the problem from outside** (rosters,
enrolment, floors), which is why items 1 and 2 outrank item 3 in value per unit of work.

## Landed today

| change | effect | where |
|---|---|---|
| `join_threshold` 0.30 → **0.50** | 87 → 28 raw speakers; end-to-end confusion 0.101 → **0.074**, worst 0.154 → 0.117 | `JOIN_THRESHOLD` |
| merge linkage single → **complete** | worst-meeting confusion 0.462 → **0.102** (chaining blow-up removed) | `MERGE_LINKAGE` |
| speaker-count hints reach both backends | the measured-good configuration becomes reachable at all | `OfflineDiarStage.diarize` |
| merge threshold decoupled from join | they answer different questions | `MERGE_THRESHOLD` |

The join threshold is the instructive one: 0.30 was calibrated for `pyannote/embedding`, the
default embedder later became TitaNet on a different cosine scale, and the constant outlived
the thing it was chosen for — silently, for months. The lesson is less about the number than
about naming constants with their calibration attached, which is now done.

## What we do next, in value order

1. **Enrol the recurring cast** from the episodes already built, and expose
   enrolment/identification in vocal-helper rather than only in notes-helper. Turns 501
   open-set problems into one closed-set problem. Free, no new model.
2. **Port FASTDIAR's encoder to ONNX** and put it behind the existing `sherpa`-style
   backend selection. MIT, CPU-only, no speaker cap, HF-free: it fits every constraint we
   have, and it is the first credible answer for the online mode.
3. **Export Nemotron-3 to ONNX/CoreML** and measure it offline *and* streaming. If the
   published numbers hold on our audio, one model replaces two paths.
4. **Make the bench report JER and |K̂−K| as first-class**, not DER alone — the *Beyond DER*
   result says a counting failure hides inside a respectable DER, and ours did.
5. **Re-measure the whole table on French broadcast audio**, not only AMI. Every number
   above is English meetings; the corpus we actually serve is French panel television.

## Honest limits of everything above

- All our clustering numbers use **reference turn boundaries**. They isolate the clustering
  rule and say nothing about VAD or segmentation error, which a real diarizer also makes.
- AMI is 4-speaker English meetings. The failure we care about is 6-plus-speaker French
  television, and the gap between those is exactly where our estimates are least trustworthy.
- `out/diar_ami`'s `offline_sherpa` 0.340 is a stale TitaNet-**small** measurement; the
  shipped path uses TitaNet-large. The table above quotes the large-model numbers, which were
  measured on two meetings, not eight. That re-run is overdue.

## Sources

- [Beyond DER: Speaker Counting in Crowded End-to-End Diarization](https://arxiv.org/abs/2609.27315)
- [FASTDIAR: Frame-level speaker encoder for Streaming Diarization](https://arxiv.org/abs/2610.02941) · [code](https://github.com/microsoft/fastdiar)
- [Streaming Sortformer (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/medennikov25_interspeech.pdf)
- [NVIDIA Nemotron 3 Diarization](https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/)
- [pyannoteAI — How Diarization Models Count Speakers (and Why It's Hard)](https://www.pyannote.ai/blog/how-diarization-models-count-speakers-(and-why-it-s-hard))
- [O-EENC-SD: Efficient Online End-to-End Neural Clustering](https://arxiv.org/pdf/2512.15229)
