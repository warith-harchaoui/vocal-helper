"""
studies/router_profile_validation
==================================

Freshly re-validate the numbers in :data:`vocal_helper.router._PROFILE` against
ground truth — **DER (quality) AND RTF (speed)** — so the aiguilleur's routing
table is anchored to a real run on this machine, not just cited from the pdbms
study.

For each path the router can pick it measures, per clip:

- **DER** via ``pyannote.metrics`` (collar 0.25, the AMI convention).
- **RTF** = diarization wall-time / audio duration (``< 1`` = faster than real
  time). Only the diarization work is timed — VAD / model load are excluded.

Paths (mirroring ``_PROFILE`` keys):

    offline_pyannote   OfflineDiarStage(backend="pyannote")   whole-buffer
    offline_nemo       OfflineDiarStage(backend="nemo")       Sortformer (short only)
    online_nemo        OnlineDiarStage(backend="nemo")        streaming embedder
    online_pyannote    OnlineDiarStage(backend="pyannote")    streaming embedder

``offline_nemo`` is **skipped on the ``ami`` (long) corpus on purpose**: the
router's own finding is that Sortformer hangs past ~25 min, so running it there
would hang this job rather than produce a number. ``sherpa`` is not re-run here
(its ONNX models are not in the local diarization-engines bundle); ADR 0002 in
``~/pasdebonneoudemauvaisesituation`` is its authoritative source.

Usage
-----
::

    PYTHONPATH=/Users/warithharchaoui/vocal-helper HF_HUB_OFFLINE=1 \\
      ~/miniconda3/bin/python studies/router_profile_validation.py \\
        --bench ~/pasdebonneoudemauvaisesituation --corpus bagarre --n 30
    ... --corpus ami --n 2

Needs the ``[pyannote]`` + ``[nemo]`` extras plus ``pyannote.metrics`` and
``soundfile``. Studies are excluded from lint / CI.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate

# Reuse the ground-truth + loading helpers from the sibling study rather than
# duplicate them (both live in studies/, which is not an importable package).
sys.path.insert(0, str(Path(__file__).parent))
from diar_der_paths import (  # noqa: E402
    COLLAR,
    SR,
    ann_from_segs,
    load_pairs,
    load_pcm,
    ref_from_rttm,
)

from vocal_helper.diar import OfflineDiarStage, OnlineDiarStage  # noqa: E402
from vocal_helper.vad import SileroVADStage  # noqa: E402


def _time_offline(stage: OfflineDiarStage, pcm: np.ndarray) -> tuple[list, float]:
    """Diarize a whole buffer, returning ``(segments, wall_seconds)``.

    Parameters
    ----------
    stage : OfflineDiarStage
        A warmed offline diarizer.
    pcm : np.ndarray
        Mono 16 kHz waveform.

    Returns
    -------
    (list, float)
        The ``(t0, t1, speaker)`` segments and the wall-clock seconds the
        ``diarize`` call took (for the RTF denominator).
    """
    # Fence the timer tightly around the diarize call: model load and VAD are
    # warmed / done elsewhere so only the diarization work feeds the RTF.
    t = time.perf_counter()
    segs = stage.diarize(pcm, SR)
    return segs, time.perf_counter() - t


def _time_online(backend: str, voiced: list, shared_emb: object) -> tuple[list, float]:
    """Run the streaming labeller over pre-VAD'd segments, timing only the diar work.

    Parameters
    ----------
    backend : str
        ``"nemo"`` or ``"pyannote"`` — the embedding backend to time.
    voiced : list
        Voiced segments from the VAD (shared across paths so VAD is not
        re-timed here).
    shared_emb : object
        A warmed embedder for ``backend`` reused across clips.

    Returns
    -------
    (list, float)
        The ``(t0, t1, speaker)`` segments and the diarization wall seconds.
    """
    stage = OnlineDiarStage(backend=backend)
    # Reuse the warm embedder + reset the clusterer so each clip starts clean.
    # Injecting the shared embedder keeps model-load out of the timed region;
    # zeroing the centroids/next_id stops one clip's speakers leaking into the
    # next clip's clustering.
    stage._embedder = shared_emb  # type: ignore[assignment]
    stage._centroids, stage._next_id = [], 0
    # Time only the streaming label pass over the (already-computed) voiced
    # segments — VAD ran upstream and is deliberately excluded from RTF.
    t = time.perf_counter()
    labelled = []
    # Feed segments through the online labeller one at a time, mimicking how
    # the live pipeline consumes captures as they arrive.
    for seg in voiced:
        s, _e = stage._label_capture(seg)
        labelled.append(s)
    wall = time.perf_counter() - t
    # Normalise the labeller's dicts down to the (t0, t1, speaker) tuple the
    # DER scorer expects.
    return [(s["t0"], s["t1"], s["speaker"]) for s in labelled], wall


async def _vad(pcm: np.ndarray, warm_vad: SileroVADStage) -> list:
    """VAD one clip using the shared warm model (imported helper wrapper)."""
    # Import here to keep the heavy sibling import lazy at module load.
    from diar_der_paths import _vad_segments

    return await _vad_segments(pcm, warm_vad)


def main() -> None:
    """Score DER + RTF for every router path on the chosen corpus and summarise."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, type=Path, help="pdbms benchmark root")
    ap.add_argument("--corpus", choices=["bagarre", "ami"], default="bagarre")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    # Resolve the (name, wav, rttm) clip triples for the chosen corpus.
    pairs = load_pairs(args.bench, args.corpus, args.n)
    # offline_nemo hangs on long meetings → only score it on the short corpus.
    score_nemo_offline = args.corpus == "bagarre"
    print(
        f"corpus={args.corpus} clips={len(pairs)} collar={COLLAR} "
        f"nemo_offline={'on' if score_nemo_offline else 'SKIP (hangs on long)'}",
        flush=True,
    )

    # Warm every backend once so model-load never pollutes the RTF numbers.
    # Each _ensure_* call forces the weights to load now, up front, outside
    # any timed region.
    off_pyannote = OfflineDiarStage(backend="pyannote")
    off_pyannote._ensure_backend()
    off_nemo = OfflineDiarStage(backend="nemo")
    # Only pay the Sortformer load cost when we will actually score it.
    if score_nemo_offline:
        off_nemo._ensure_backend()
    warm_nemo = OnlineDiarStage(backend="nemo")
    warm_nemo._ensure_embedder()
    warm_pyan = OnlineDiarStage(backend="pyannote")
    warm_pyan._ensure_embedder()
    # One shared VAD model feeds both streaming paths (see the shared pass
    # below), so warm it once here too.
    warm_vad = SileroVADStage()
    warm_vad._ensure_model()

    # Accumulators: per-path DER list + total (wall, audio) seconds for RTF.
    ders: dict[str, list[float]] = {
        "offline_pyannote": [],
        "offline_nemo": [],
        "online_nemo": [],
        "online_pyannote": [],
    }
    # walls / audio accumulate the RTF numerator (diar wall) and denominator
    # (audio seconds) so the ratio is computed over the whole corpus, not
    # averaged per clip.
    walls: dict[str, float] = {k: 0.0 for k in ders}
    audio: dict[str, float] = {k: 0.0 for k in ders}

    t_start = time.perf_counter()
    for idx, (name, wav, rttm) in enumerate(pairs):
        # Per-clip try/except: a single bad file should skip that clip, not
        # abort the whole corpus sweep.
        try:
            pcm = load_pcm(wav)
            dur = len(pcm) / SR
            ref = ref_from_rttm(rttm)

            # --- offline pyannote (the robust default) ---
            # Score DER against ground truth and bank wall + audio for RTF.
            segs, wall = _time_offline(off_pyannote, pcm)
            ders["offline_pyannote"].append(
                DiarizationErrorRate(collar=COLLAR)(ref, ann_from_segs(segs))
            )
            walls["offline_pyannote"] += wall
            audio["offline_pyannote"] += dur

            # --- offline nemo (short only) ---
            # Guarded by score_nemo_offline because Sortformer hangs on long
            # meetings (see module docstring).
            if score_nemo_offline:
                segs, wall = _time_offline(off_nemo, pcm)
                ders["offline_nemo"].append(
                    DiarizationErrorRate(collar=COLLAR)(ref, ann_from_segs(segs))
                )
                walls["offline_nemo"] += wall
                audio["offline_nemo"] += dur

            # --- streaming paths share one VAD pass (VAD time excluded from RTF) ---
            # Run VAD once and hand the same voiced segments to both online
            # backends so VAD cost is charged to neither path's RTF.
            voiced = asyncio.run(_vad(pcm, warm_vad))
            for backend, key, emb in (
                ("nemo", "online_nemo", warm_nemo._embedder),
                ("pyannote", "online_pyannote", warm_pyan._embedder),
            ):
                # Same score-and-bank pattern as the offline paths, but on the
                # streaming labeller timed by _time_online.
                segs, wall = _time_online(backend, voiced, emb)
                ders[key].append(DiarizationErrorRate(collar=COLLAR)(ref, ann_from_segs(segs)))
                walls[key] += wall
                audio[key] += dur

            print(
                f"[{idx + 1}/{len(pairs)}] {name:14s} "
                f"pyan={ders['offline_pyannote'][-1]:.3f} "
                f"onl_pyan={ders['online_pyannote'][-1]:.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx + 1}/{len(pairs)}] {name}: SKIP {exc!r}", flush=True)

    # Summary: DER (quality) + RTF (speed) per path, to reconcile with _PROFILE.
    print(f"\n=== {args.corpus} DER+RTF ({time.perf_counter() - t_start:.0f}s) ===", flush=True)
    print(f"  {'path':18s} {'n':>3s}  {'DER_mean':>8s} {'DER_med':>8s}  {'RTF':>7s}", flush=True)
    for key, vals in ders.items():
        # Skip paths with no scores (e.g. offline_nemo on the ami corpus,
        # which was intentionally not run).
        if vals:
            a = np.array(vals)
            # Report both mean and median DER: the median resists a single
            # pathological clip, the mean shows the tail. RTF is the pooled
            # ratio (guarded against a zero denominator).
            rtf = walls[key] / audio[key] if audio[key] > 0 else float("nan")
            print(
                f"  {key:18s} {len(a):3d}  {a.mean():8.3f} {np.median(a):8.3f}  {rtf:7.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
