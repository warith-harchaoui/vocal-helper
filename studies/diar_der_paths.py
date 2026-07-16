"""
studies/diar_der_paths
======================

Anchor vocal-helper's diarization **default** to real DER. Runs the three
diarization paths the package ships against ground-truth RTTM and scores them
with ``pyannote.metrics`` (collar 0.25, the AMI convention):

    offline_pyannote  -> OfflineDiarStage(backend="pyannote")   whole-buffer
    online_baseline   -> VAD -> OnlineDiarStage(nemo)           no refine
    online_refine     -> VAD -> OnlineDiarStage(nemo) + refine_on_close

Result (2026-07-16, this machine; ``studies`` are excluded from lint/CI):

    corpus                 offline   online_baseline   online_refine
    ---------------------  --------  ----------------  --------------
    AMI (2 real meetings)   0.122        0.497             0.351
    bagarre (40 clips)      0.338        0.586             0.592

Reading it: offline pyannote is literature-grade (Bredin 2023 ~0.188
uncollared); ``refine_on_close`` roughly halves the online DER on meetings
that over-segment (ES2011a 0.588 -> 0.296) and never hurts; but the online
path is still ~3x the offline DER because it cannot model overlapped speech.
Hence ``vocal-helper file --no-real-time`` auto-selects offline when the
bundle is present, and batch integrators should use OfflineDiarStage /
OfflinePipeline. See ``vocal_helper/diar.py`` module docstring.

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
    """Return ``[(name, wav, rttm), …]`` for the requested corpus."""
    if corpus == "bagarre":
        wavs = sorted(glob.glob(str(bench / "data" / "bagarre" / "mix_*.wav")))[:n]
        return [(Path(w).stem, Path(w), Path(w).with_suffix(".rttm")) for w in wavs]
    if corpus == "ami":
        out = []
        for d in sorted((bench / "data" / "ami" / "dev-slice").iterdir()):
            wav, rttm = d / "mix.wav", d / "words.rttm"
            if wav.exists() and rttm.exists():
                out.append((d.name, wav, rttm))
        return out[:n]
    raise SystemExit(f"unknown corpus {corpus!r}")


def load_pcm(wav: Path) -> np.ndarray:
    """Load a WAV as mono float32 at 16 kHz."""
    pcm, sr = sf.read(str(wav))
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sr != SR:
        import scipy.signal as ss

        pcm = ss.resample(pcm, int(len(pcm) * SR / sr))
    return pcm.astype(np.float32)


def ref_from_rttm(rttm: Path) -> Annotation:
    """Word-level RTTM -> reference Annotation (same-speaker words merged, gap<=0.3s)."""
    rows = []
    for line in rttm.read_text().splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            rows.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    rows.sort()
    ann = Annotation()
    turns: dict[str, tuple[float, float]] = {}
    for t0, t1, spk in rows:
        if spk in turns and t0 <= turns[spk][1] + 0.3:
            turns[spk] = (turns[spk][0], max(turns[spk][1], t1))
        else:
            if spk in turns:
                ann[Segment(*turns[spk])] = spk
            turns[spk] = (t0, t1)
    for spk, (a, b) in turns.items():
        ann[Segment(a, b)] = spk
    return ann


def ann_from_segs(segs: list[tuple[float, float, str]]) -> Annotation:
    """Build an Annotation from ``[(t0, t1, speaker), …]``."""
    ann = Annotation()
    for i, (t0, t1, spk) in enumerate(segs):
        if t1 > t0:
            ann[Segment(t0, t1), i] = spk
    return ann


async def _vad_segments(pcm: np.ndarray, warm_vad: SileroVADStage) -> list:
    """VAD one clip with a fresh stage that reuses the warm model."""
    vad = SileroVADStage()
    vad._model, vad._torch = warm_vad._model, warm_vad._torch
    inbox: asyncio.Queue = asyncio.Queue(maxsize=256)
    outbox: asyncio.Queue = asyncio.Queue()
    segs: list = []

    async def feed():
        for i in range(0, len(pcm), FRAME):
            await inbox.put(PcmFrame(t0=i / SR, sample_rate=SR, pcm=pcm[i : i + FRAME]))
        await inbox.put(None)

    async def drain():
        while True:
            item = await outbox.get()
            if item is None:
                return
            segs.append(item)

    await asyncio.gather(feed(), vad.run(inbox, outbox), drain())
    return segs


def main() -> None:
    """Score the three paths on the chosen corpus and print a DER summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, type=Path, help="pdbms benchmark root")
    ap.add_argument("--corpus", choices=["bagarre", "ami"], default="bagarre")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    pairs = load_pairs(args.bench, args.corpus, args.n)
    print(f"corpus={args.corpus}  clips={len(pairs)}  collar={COLLAR}")

    offline = OfflineDiarStage(backend="pyannote")
    offline._ensure_backend()
    warm_online = OnlineDiarStage(backend="nemo")
    warm_online._ensure_embedder()
    warm_vad = SileroVADStage()
    warm_vad._ensure_model()
    shared_emb = warm_online._embedder

    ders: dict[str, list[float]] = {
        "offline_pyannote": [],
        "online_baseline": [],
        "online_refine": [],
    }
    t0 = time.perf_counter()
    for idx, (name, wav, rttm) in enumerate(pairs):
        try:
            pcm = load_pcm(wav)
            ref = ref_from_rttm(rttm)
            hyp_off = ann_from_segs(offline.diarize(pcm, SR))
            voiced = asyncio.run(_vad_segments(pcm, warm_vad))
            stage = OnlineDiarStage(backend="nemo", refine_on_close=True)
            stage._embedder = shared_emb
            stage._centroids, stage._next_id = [], 0
            labelled, embs = [], []
            for seg in voiced:
                s, e = stage._label_capture(seg)
                labelled.append(s)
                embs.append(e)
            hyp_base = ann_from_segs([(s["t0"], s["t1"], s["speaker"]) for s in labelled])
            refined = stage._refine_labels(labelled, embs)
            hyp_ref = ann_from_segs(
                [(s["t0"], s["t1"], lab) for lab, s in zip(refined, labelled)]
            )
            for key, hyp in [
                ("offline_pyannote", hyp_off),
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
            print(f"[{idx + 1}/{len(pairs)}] {name}: SKIP {exc!r}", flush=True)

    print(f"\n=== {args.corpus} DER summary ({time.perf_counter() - t0:.0f}s) ===")
    for key, vals in ders.items():
        if vals:
            a = np.array(vals)
            print(f"  {key:18s} n={len(a):3d}  mean={a.mean():.3f}  median={np.median(a):.3f}")


if __name__ == "__main__":
    main()
