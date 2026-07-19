"""
studies/diar_der_paths
======================

Anchor vocal-helper's diarization **default** to real DER. Runs the three
diarization paths the package ships against ground-truth RTTM and scores them
with ``pyannote.metrics`` (collar 0.25, the AMI convention):

    offline_pyannote  -> OfflineDiarStage(backend="pyannote")   whole-buffer
    offline_nemo      -> OfflineDiarStage(backend="nemo")       Sortformer
    online_baseline   -> VAD -> OnlineDiarStage(nemo)           no refine
    online_refine     -> VAD -> OnlineDiarStage(nemo) + refine_on_close

Result (2026-07-16, this machine; ``studies`` are excluded from lint/CI):

    corpus                 offline_pyannote  offline_nemo  online_base  online_ref
    ---------------------  ----------------  ------------  -----------  ----------
    AMI (2 real meetings)      0.122            0.242          0.497        0.351
    bagarre (40 clips)         0.338            0.177          0.586        0.592

Reading it — a length crossover on the offline backends: pyannote is
literature-grade (Bredin 2023 ~0.188 uncollared) and wins on long meetings
(whole-buffer, no speaker cap); NeMo Sortformer (``diar_sortformer_4spk-v1``,
end-to-end + overlap-aware but 4-speaker / ~90 s capped) nearly halves the DER
on short <=4-speaker clips yet degrades once it must chunk long audio. So
pyannote stays the offline default and Sortformer is the pick for short
<=4-speaker workloads. ``refine_on_close`` roughly halves the *online* DER on
meetings that over-segment (ES2011a 0.588 -> 0.296) and never hurts, but the
online path is still ~3x the offline DER. Hence ``vocal-helper file
--no-real-time`` auto-selects offline pyannote when the bundle is present, and
batch integrators should use OfflineDiarStage / OfflinePipeline. See
``vocal_helper/diar.py`` module docstring.

Data (off-repo, from the pdbms benchmark):
    bagarre : <bench>/data/bagarre/mix_*.wav + mix_*.rttm   (synthetic 3-spk)
    AMI     : <bench>/data/ami/dev-slice/<meeting>/{mix.wav, words.rttm}

Usage
-----
::

    python studies/diar_der_paths.py --bench ~/pasdebonneoudemauvaisesituation \\
        --corpus bagarre --n 40
    python studies/diar_der_paths.py --bench ~/... --corpus ami --n 2

Needs the ``[pyannote]`` + ``[nemo]`` extras plus ``pyannote.metrics`` and
``soundfile`` (ground-truth tooling, not a runtime dependency).

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from vocal_helper.diar import OfflineDiarStage, OnlineDiarStage
from vocal_helper.types import PcmFrame
from vocal_helper.vad import SileroVADStage

SR = 16_000
FRAME = 320
COLLAR = 0.25


def load_pairs(bench: Path, corpus: str, n: int) -> list[tuple[str, Path, Path]]:
    """Return ``[(name, wav, rttm), …]`` for the requested corpus.

    Parameters
    ----------
    bench : Path
        Root of the off-repo pdbms benchmark tree.
    corpus : str
        Either ``"bagarre"`` (synthetic 3-speaker mixes) or ``"ami"``
        (real meeting dev-slice).
    n : int
        Maximum number of clips to return.

    Returns
    -------
    list of tuple of (str, Path, Path)
        Up to ``n`` ``(name, wav, rttm)`` triples, each pairing a mix
        clip with its ground-truth RTTM.

    Raises
    ------
    SystemExit
        If ``corpus`` is neither ``"bagarre"`` nor ``"ami"``.
    """
    # bagarre: flat directory of mix_*.wav, each with a sibling .rttm.
    if corpus == "bagarre":
        wavs = sorted(glob.glob(str(bench / "data" / "bagarre" / "mix_*.wav")))[:n]
        return [(Path(w).stem, Path(w), Path(w).with_suffix(".rttm")) for w in wavs]
    # ami: one sub-dir per meeting holding mix.wav + words.rttm.
    if corpus == "ami":
        out = []
        for d in sorted((bench / "data" / "ami" / "dev-slice").iterdir()):
            wav, rttm = d / "mix.wav", d / "words.rttm"
            # Only keep meetings where both files are present.
            if wav.exists() and rttm.exists():
                out.append((d.name, wav, rttm))
        return out[:n]
    # Any other corpus name is a caller mistake — stop loudly.
    raise SystemExit(f"unknown corpus {corpus!r}")


def load_pcm(wav: Path) -> np.ndarray:
    """Load a WAV as mono float32 at 16 kHz.

    Parameters
    ----------
    wav : Path
        Path to the mix clip.

    Returns
    -------
    numpy.ndarray
        Mono ``float32`` samples resampled to :data:`SR` (16 kHz) — the
        rate every diarization backend in this study expects.
    """
    pcm, sr = sf.read(str(wav))
    # Downmix to mono; the diarizers work on a single channel.
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    # Resample only when needed — scipy is imported lazily so clips
    # already at 16 kHz never pull in the dependency.
    if sr != SR:
        import scipy.signal as ss

        pcm = ss.resample(pcm, int(len(pcm) * SR / sr))
    return pcm.astype(np.float32)


def ref_from_rttm(rttm: Path) -> Annotation:
    """Word-level RTTM -> reference Annotation (same-speaker words merged, gap<=0.3s).

    Parameters
    ----------
    rttm : Path
        Word-level ground-truth RTTM file.

    Returns
    -------
    pyannote.core.Annotation
        The reference diarization: adjacent words of the same speaker
        separated by no more than 0.3 s are merged into one turn, which
        keeps the reference from over-fragmenting into per-word segments.
    """
    # Extract (start, end, speaker) rows from every SPEAKER line.
    rows = []
    for line in rttm.read_text().splitlines():
        f = line.split()
        # Need at least 8 fields so f[7] (the speaker gid) is present.
        if len(f) >= 8 and f[0] == "SPEAKER":
            rows.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    # Time order so the per-speaker gap test below sees words in sequence.
    rows.sort()
    ann = Annotation()
    # Track the open turn per speaker so consecutive close words can be
    # extended in place instead of emitting a segment each.
    turns: dict[str, tuple[float, float]] = {}
    for t0, t1, spk in rows:
        # Same speaker within the 0.3 s bridge: extend the open turn.
        if spk in turns and t0 <= turns[spk][1] + 0.3:
            turns[spk] = (turns[spk][0], max(turns[spk][1], t1))
        else:
            # Gap too large (or new speaker): flush the previous open turn
            # for this speaker, then start a fresh one.
            if spk in turns:
                ann[Segment(*turns[spk])] = spk
            turns[spk] = (t0, t1)
    # Emit whatever open turn each speaker ended on.
    for spk, (a, b) in turns.items():
        ann[Segment(a, b)] = spk
    return ann


def ann_from_segs(segs: list[tuple[float, float, str]]) -> Annotation:
    """Build an Annotation from ``[(t0, t1, speaker), …]``.

    Parameters
    ----------
    segs : list of tuple of (float, float, str)
        Hypothesis segments as ``(start_s, end_s, speaker)``.

    Returns
    -------
    pyannote.core.Annotation
        A pyannote Annotation carrying one labelled segment per input
        tuple, ready to be scored against the reference.
    """
    ann = Annotation()
    # Index each segment (the enumerate i) so identical (t0, t1) spans
    # with different labels don't collide as track keys.
    for i, (t0, t1, spk) in enumerate(segs):
        # Drop zero/negative-length spans; pyannote rejects empty segments.
        if t1 > t0:
            ann[Segment(t0, t1), i] = spk
    return ann


async def _vad_segments(pcm: np.ndarray, warm_vad: SileroVADStage) -> list:
    """VAD one clip with a fresh stage that reuses the warm model.

    Parameters
    ----------
    pcm : numpy.ndarray
        Mono 16 kHz ``float32`` samples for the whole clip.
    warm_vad : SileroVADStage
        A stage whose Silero model is already loaded; its model/torch
        handles are borrowed so we pay the load cost only once.

    Returns
    -------
    list
        The voiced segments the stage emitted, in arrival order.

    Notes
    -----
    A fresh :class:`SileroVADStage` per clip resets the stage's streaming
    state (so clips don't bleed into each other) while sharing the heavy
    model instance for speed.
    """
    # New stage for clean per-clip streaming state; graft the warm model
    # and torch handle so we skip a second model load.
    vad = SileroVADStage()
    vad._model, vad._torch = warm_vad._model, warm_vad._torch
    # Bounded inbox applies back-pressure so feed() can't outrun the stage;
    # unbounded outbox just collects results.
    inbox: asyncio.Queue = asyncio.Queue(maxsize=256)
    outbox: asyncio.Queue = asyncio.Queue()
    segs: list = []

    async def feed() -> None:
        """Stream the clip into the stage as fixed-size frames, then stop.

        Returns
        -------
        None
        """
        # Chop the clip into FRAME-sample PcmFrames carrying their start
        # time so the stage can timestamp voiced runs correctly.
        for i in range(0, len(pcm), FRAME):
            await inbox.put(PcmFrame(t0=i / SR, sample_rate=SR, pcm=pcm[i : i + FRAME]))
        # Sentinel None tells the stage the input stream is done.
        await inbox.put(None)

    async def drain() -> None:
        """Collect emitted voiced segments until the stage signals end.

        Returns
        -------
        None
        """
        # Pull results until the stage's None sentinel closes the outbox.
        while True:
            item = await outbox.get()
            if item is None:
                return
            segs.append(item)

    # Run producer, stage, and consumer concurrently so the clip streams
    # through in one pass.
    await asyncio.gather(feed(), vad.run(inbox, outbox), drain())
    return segs


def main() -> None:
    """Score the three paths on the chosen corpus and print a DER summary.

    Loads the requested corpus, warms every diarization backend once,
    then for each clip computes the DER of the two offline paths and the
    two online paths against the ground truth, and prints per-clip lines
    plus a pooled mean/median summary.

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, type=Path, help="pdbms benchmark root")
    ap.add_argument("--corpus", choices=["bagarre", "ami"], default="bagarre")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    # Resolve the clip list for the chosen corpus up front.
    pairs = load_pairs(args.bench, args.corpus, args.n)
    print(f"corpus={args.corpus}  clips={len(pairs)}  collar={COLLAR}")

    # Warm every backend once so the per-clip loop times inference, not
    # model loading. Two offline diarizers (pyannote + NeMo Sortformer),
    # one online stage whose embedder we share, and one Silero VAD.
    offline = OfflineDiarStage(backend="pyannote")
    offline._ensure_backend()
    offline_nemo = OfflineDiarStage(backend="nemo")
    offline_nemo._ensure_backend()
    warm_online = OnlineDiarStage(backend="nemo")
    warm_online._ensure_embedder()
    warm_vad = SileroVADStage()
    warm_vad._ensure_model()
    # Reuse this one embedder across every per-clip online stage below.
    shared_emb = warm_online._embedder

    # One DER list per scored path; medians/means are taken at the end.
    ders: dict[str, list[float]] = {
        "offline_pyannote": [],
        "offline_nemo": [],
        "online_baseline": [],
        "online_refine": [],
    }
    t0 = time.perf_counter()
    for idx, (name, wav, rttm) in enumerate(pairs):
        # A per-clip failure (missing file, backend error) shouldn't abort
        # the sweep — log a SKIP and move on.
        try:
            # Shared inputs: audio + reference annotation for this clip.
            pcm = load_pcm(wav)
            ref = ref_from_rttm(rttm)
            # Offline paths run the whole buffer through each backend.
            hyp_off = ann_from_segs(offline.diarize(pcm, SR))
            hyp_off_nemo = ann_from_segs(offline_nemo.diarize(pcm, SR))
            # Online paths start from VAD-carved voiced segments.
            voiced = asyncio.run(_vad_segments(pcm, warm_vad))
            # Fresh online stage per clip (clean clustering state) but with
            # the shared embedder; reset centroids/id counter explicitly.
            stage = OnlineDiarStage(backend="nemo", refine_on_close=True)
            stage._embedder = shared_emb
            stage._centroids, stage._next_id = [], 0
            # Label each voiced segment online, keeping its embedding so
            # the refine pass below can re-cluster with full context.
            labelled, embs = [], []
            for seg in voiced:
                s, e = stage._label_capture(seg)
                labelled.append(s)
                embs.append(e)
            # Baseline online hypothesis: the streaming labels as-emitted.
            hyp_base = ann_from_segs([(s["t0"], s["t1"], s["speaker"]) for s in labelled])
            # Refined hypothesis: re-label using all embeddings at once
            # (the refine_on_close pass), keeping each segment's timing.
            refined = stage._refine_labels(labelled, embs)
            hyp_ref = ann_from_segs([(s["t0"], s["t1"], lab) for lab, s in zip(refined, labelled)])
            # Score all four hypotheses against the same reference; a new
            # metric object per call keeps collar accounting independent.
            for key, hyp in [
                ("offline_pyannote", hyp_off),
                ("offline_nemo", hyp_off_nemo),
                ("online_baseline", hyp_base),
                ("online_refine", hyp_ref),
            ]:
                ders[key].append(DiarizationErrorRate(collar=COLLAR)(ref, hyp))
            print(
                f"[{idx + 1}/{len(pairs)}] {name:14s} "
                f"off={ders['offline_pyannote'][-1]:.3f} "
                f"base={ders['online_baseline'][-1]:.3f} "
                f"refine={ders['online_refine'][-1]:.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            # Keep going on any per-clip error; the summary just excludes it.
            print(f"[{idx + 1}/{len(pairs)}] {name}: SKIP {exc!r}", flush=True)

    # Pooled summary: mean + median DER per path over the clips that scored.
    print(f"\n=== {args.corpus} DER summary ({time.perf_counter() - t0:.0f}s) ===")
    for key, vals in ders.items():
        # Skip paths with no successful clips.
        if vals:
            a = np.array(vals)
            print(f"  {key:18s} n={len(a):3d}  mean={a.mean():.3f}  median={np.median(a):.3f}")


if __name__ == "__main__":
    main()
