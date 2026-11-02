# Diarization: what we measured, and what we now ship

*2026-11-02. Companion to [`diarization-roadmap.md`](diarization-roadmap.md), which set out the
problem on 2026-10-20. This one answers it with numbers, including the answers that turned out
to be wrong.*

## Where we started, and where we are

| | free local, live | free local, offline |
|---|---:|---:|
| **before** | 0.673 error, **1 label** for 4 speakers | 0.299 error, 5 labels |
| **after** | **0.090**, 4 labels, 0.000 confusion | **0.090**, 4 labels |

Same engine in both modes: **Nemotron 3 Diarization**, through its ONNX graph, with no torch on
the decision path, no HuggingFace at runtime and no token. The weights travel in the
diarization-engines bundle the project already serves.

Against a **paid** reference — French television episodes already diarized by pyannoteAI, read
from the wallet's cache — on the only three episodes where that service finds eight speakers or
fewer:

| engine | episode 1 *(17 min, 8 paid speakers)* | episode 2 *(20 min, 7)* |
|---|---:|---:|
| **nemotron** | **0.017** *(8 labels)* | **0.072** *(6)* |
| pyannote community-1 | 0.059 *(7)* | 0.155 *(7)* |
| nemo sortformer | 0.128 *(10)* | 0.108 *(13)* |
| sherpa | 0.447 *(29)* | 0.551 *(**102**)* |

Read that number for what it is: a **disagreement with a paid service**, not an absolute error
rate. pyannoteAI is not the truth — its own error rate is in the 0.10–0.20 range, and two engines
can differ from it while both being right. It is, however, the question someone asks before
deciding to pay.

## The roadmap's diagnosis was right: counting, not DER

The roadmap said the problem was not that the error rate was too high but that **nothing could
count speakers**, and that a duration-weighted metric hides exactly that. Everything measured
since agrees, and the label column was the most informative one in every table. Two illustrations:

- The greedy online clusterer that shipped as the live default fails in **both directions**
  depending on the material: one label for four speakers on clean synthetic speech, forty-four to
  sixty-eight on real meetings. No threshold can fix that, because the value that prevents the
  explosion causes the collapse.
- sherpa, the torch-free path, emits **102 labels for 7 speakers** on real television audio. It is
  last in the preference order, but its documentation presented it as "portable and slightly
  worse", which is false: on real material it is unusable.

## Four threshold parameterisations, four refutations

Before reaching for a different model, we tried to make the existing clusterer hold identity.
Every attempt was a way of choosing *when two voices are the same voice*:

| parameterisation | why it should work | what measurement said |
|---|---|---|
| absolute cosine threshold | simplest | 0.8 is optimal on one meeting and merges two real speakers on another |
| fraction of the median inter-speaker distance | the room changes the scale of distances | both meetings have the **same** median, 0.834 and 0.847 |
| fraction of a low quantile | the confusable pair is what decides | separates them correctly (0.447 vs 0.287) but over-corrects twofold |
| position between the intra- and inter-speaker scales | no hand-chosen constant left | spread of 0.213 against 0.150 for the raw thresholds |

On three meetings the optimal thresholds are 0.750 / 0.600 / 0.600 and the corresponding
normalised positions 0.879 / 0.666 / 0.687. **Normalising widens the spread.** In hindsight this
was the clue: we were tuning a decision a model should be making. Nemotron has no threshold — it
carries an arrival-ordered speaker cache and *learns* identity.

The measurements kept are in `studies/diar_incremental_delta_sweep.py` and
`studies/diar_incremental_scale_calibration.py`, with the refutation written into the module so
nobody retries it blind.

## Centroid-free clustering: a closed branch

A speaker's centroid is a mean, therefore a **point**, and a point cannot represent a spread —
which is what one voice near and far from a microphone actually is. So we replaced it by a bounded
set of exemplars, matching on the distance to its **best point**. Four reference rules and five
merge thresholds later, the centroid wins on every metric including the worst case.

The mean turns out to be a **regulariser** we had underestimated, and the best point is an
*optimistic* estimator: a minimum over a set shrinks mechanically as the set grows, so one noisy
exemplar is enough to capture a speaker. Asking for the k-th nearest instead of the nearest
recovers most of the gap (0.529 → 0.363) without closing it. `vocal_helper/exemplar.py` keeps the
implementation and the verdict; it is not the default and should not become one without new
evidence.

## Identity can travel by time alone

A mechanism worth recording because it works and because it produced two real bug fixes. The
current chunk is diarized **twice**: once alone, and once preceded by a synthetic primer in which
every known speaker occupies a slot at a *known* position. Reading which cluster covers which slot
names the clusters; temporal overlap then carries those names to the first diarization. No
embedding geometry anywhere.

Two measurements framed it. Chunking costs little — 0.025 to 0.051 of error — so working chunk by
chunk is legitimate. But naming cost 0.111, remarkably stable across meetings, therefore a single
cause. It was coarse: **references were frozen at first sight**, so the first seconds ever heard of
a speaker defined them for the whole session, including when harvested from an already-merged
cluster. Unfrozen, the naming loss falls to **0.003** — the mechanism reaches its own ceiling.

And the primer read-back was a greedy argmax where it is a **bijection**. Whenever the engine
merged two primer speakers, one label won two slots and the second write **overwrote** the first:
a known speaker silently lost its label and came back as a newcomer. The scheme invented speakers
exactly where it should have recognised them. It is an optimal assignment now, and
`read_back_identities` explains why in the module.

## Offline quality, live: the growing buffer

Push that logic to its conclusion and the primer becomes unnecessary. It exists for one case — a
speaker absent from the region the two diarizations share. If the second pass covers **all the
audio since the start**, that case disappears: whoever has spoken is in the shared region, which is
then nearly the whole signal.

The last pass is therefore a whole-file diarization, so the final answer **is** the offline one.
Measured on the eight AMI dev-slice meetings: 0.176 median, the right speaker count, to the third
decimal. Geometric spacing keeps the cost proportional to the session rather than to its square —
2.6 to 3.3 times the audio, a real-time factor of 0.10 to 0.16.

What it costs is not quality but **freshness**: speech after the last pass has no authoritative
label yet, and at `ratio=1.6` that wait is 0.375 times the session — fine for twenty minutes, not
for two hours, where `max_period_s` bounds it. `voh.GrowingBufferDiarStage` ships it, and it
remains the best path that needs no extra model and the only one measured on a real recording with
a human reference.

## AMI cannot arbitrate, and we proved it rather than assumed it

Nemotron ranks **last** on AMI — 0.487 against pyannote's 0.192 — while showing a speaker
confusion of 0.003 against 0.036. That does not hold together, and the whole gap is speech it does
not emit: 616 seconds of the 815 annotated. Four explanations were tested:

| candidate | test | verdict |
|---|---|---|
| extraction threshold | swept to 0.05 | missed only falls 0.461 → 0.359 while false alarm triples |
| signal level | RMS raised 2.8× | missed gets slightly **worse** |
| reverberation | 0.3 s and 0.6 s RT60 | missed stays flat: 0.090 / 0.093 / 0.101 |
| **the annotation** | signal energy where it is missed | **21.8 dB below** emitted speech, 5.4 dB above annotated silence |

The missed speech sits at the room microphone's noise floor. AMI's reference is transcribed from
the **headset** microphones while the engine is given the room mix, so it marks words the signal
does not carry — and pyannote has AMI in its training data. Both biases point the same way and
neither is separable on that corpus. The corollary is less flattering for pyannote: it emits into
that noise floor, which the biased reference rewards and an exact one punishes — on the
reverberated mix its missed detection is 0.296 against Nemotron's 0.101.

`studies/diar_far_field.py` and the paid-reference study carry those measurements.

## The latency knob, and a default we got wrong first

Latency is one setting, in seconds; the schedule follows. The first default took the longest viable
context — 6.4 s — because more context is **cheaper** at equal quality: a real-time factor of 0.166
against 0.258 for 0.092 error against 0.090.

That equality only exists in an **anechoic** room, and a meeting room is not one. Reverberated at
0.6 s RT60 the two separate clearly:

| latency | error | confusion | ≥300 s agreement |
|---|---:|---:|---:|
| 6.40 s | 0.350 | 0.190 | 0.372 |
| **1.04 s** | **0.252** | **0.151** | **0.668** |

Long-range agreement is nearly doubled. Long context dilutes the decision once the room smears the
signal: more material, but more confused material. The default is therefore the schedule the
provenance publishes and validated — which should have been the prior.

`discussion-helper` keeps `diar_latency_s=1.04`; a clean close-microphone recording can ask for
`latency_s=None` and save a third of the compute.

## Two ceilings that are not ours to lift

**Eight speakers.** On an hour of panel television the paid reference finds thirteen to forty-two
speakers — panel, reports, archive, voice-over. Nemotron caps at eight channels and is therefore
structurally out of the running on that material. For FranceTV-style work in batch, pyannote
remains the only free engine that does not hit a count ceiling.

**The graph's offline schedule does not run.** Its export declares the 30.4 s schedule the only one
verified end to end. It fails:

```
Reshape '/pos_enc/Reshape_3' : Input shape {5000,64}, requested shape {1,1,5144,64}
```

The first reading was a sequence-length limit — 625 frames for `[cache | fifo | chunk]`. That is
not the whole story, and a bisection probe on the file itself accepted 321 chunk frames, which is
what put us on the right track. The real cause: **the fifo grows without bound.** The update adds
`chunk_len` frames to it and removes only `fifo_len`, so a chunk longer than the fifo makes it
grow — at 281 chunk frames it reaches 241 instead of 40 and the sequence hits 826. This is a
faithful port of the reference, whose rule only holds when the chunk does not exceed the fifo: its
three published streaming schedules respect that (9, 6, 3 frames) and its offline schedule, at 340,
violates it. Usable context therefore tops out at 6.4 s, and the limit is checked rather than
assumed.

## Execution providers: measured, not reasoned

The first default put **CoreML** first, by the obvious argument: ONNX Runtime has no Metal
provider, so on Apple Silicon acceleration goes through CoreML. The reasoning was sound and the
conclusion false. CoreML takes only 1193 of 2827 nodes and splits them into **316 partitions**, so
each streaming step pays that many crossings: a real-time factor of 1.257, i.e. **slower than real
time**, against 0.528 for CPU alone.

Forcing eight threads on an eight-performance-core machine was equally counterproductive: 0.953
against 0.528 when ONNX Runtime chooses. Both "optimisations" hurt. CPU leads the order now, CUDA
ahead of it on operator coverage grounds — unmeasured, and the comment says so.

## Does Rust buy real-time factor? No. It buys places to run.

Profiled on two minutes of audio, with the graph call and the mel front end timed separately:

| | time | share |
|---|---:|---:|
| ONNX graph *(C++, identical from Rust)* | 90.47 s | **99.8 %** |
| log-mel front end *(numpy, FFT in C)* | 0.06 s | 0.1 % |
| cache bookkeeping + Python | 0.10 s | 0.1 % |

A Rust port's ceiling is **0.2 % of the time**: 0.754 instead of 0.755. The `ort` crate calls the
same C++ library, and the two Python pieces are either vectorised or operating on 264×512 arrays
about once a second.

Rust is still justified, for a different reason, and it should be announced as such: **on mobile
the Python path does not exist.** One does not reasonably embed CPython and a scientific stack in
an iOS or Android application. That is not a performance argument but a *possibility* one — plus a
single binary per platform, no dependency resolution at install, and no interpreter to start. Same
engine, same speed, places to run that Python does not reach. Promising an RTF gain would be
false, and the measurement says so.

## What is not measured

- **Beyond eight speakers**, which is Nemotron's declared ceiling and therefore the most
  interesting case. Its row on an eight-speaker mix did not complete.
- **Real recordings beyond three episodes.** Everything favouring Nemotron rests on synthetic
  mixes plus three television episodes; the one classic corpus available is the one whose
  annotation is the problem.
- **CUDA**, for want of an NVIDIA machine.
- **Real-time factors have a twofold spread** depending on machine load — 0.258 then 0.755 for the
  same schedule. The figures quoted here are orders of magnitude, not specifications, and a clean
  re-measurement on an idle machine is owed.
