"""Diar embedding backend — pyannote/embedding vs TitaNet (NeMo).

Goal
----
``OnlineDiarStage`` ships with ``backend='pyannote'`` by default
(reuses ``pyannote/embedding``). The pdbms canonical study
(§10.5, N=2089 cells) showed NeMo's TitaNet has a wider cosine
distribution and better separation on noisy / short clips — but the
diarization pipeline there uses NeMo end-to-end, not just the
embedding. This study isolates the **embedding choice** for our
per-voiced-segment online diarizer.

Protocol
--------

- corpus  : AMI dev-slice ``mix.wav`` (full meeting, 16 kHz mono).
- diarize : truth from ``words.rttm`` bridged at 200 ms — same
  protocol as the pdbms ideal-duration sweep.
- VAD     : Silero v5 (cadence 48 ms, threshold 0.5) — same as the
  pipeline default.
- per voiced segment :
    1. compute ``backend.embed(audio_segment)`` for both backends ;
    2. label the segment with the ground-truth speaker ;
    3. accumulate per-speaker embedding lists.
- once the meeting is fully consumed :
    - intra-speaker cosine distance : median over (e_i, e_j) pairs of
      the same speaker. Lower = tighter cluster.
    - inter-speaker cosine distance : median over (e_i, e_j) pairs of
      different speakers. Higher = more separable.
    - **separability margin** = inter − intra. Higher is better.
- the backend with the larger margin is the better fit for
  cosine-distance running-mean clustering.

We also report median wall-time per embed call (a quick RTF proxy).

Why this measure (and not DER end-to-end)
-----------------------------------------
The online diarizer's only job downstream of the embedding is "is
this segment closer to centroid A or centroid B ?". The embedding
that maximises the inter-vs-intra margin is, by definition, the one
that minimises misclustering errors. End-to-end DER is dominated by
VAD / segmentation noise that the embedding does not control, so the
margin is a cleaner ablation signal.

Author : Warith HARCHAOUI — 2026-06-30
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Pull the same helpers the production diarizer uses.
from vocal_helper.diar import _PyannoteEmbedder, _TitaNetEmbedder
from vocal_helper._settings import resolve_hf_token

# pdbms's VAD utilities — already imported by the pipeline.
import sys
sys.path.insert(
    0,
    "/Users/warithharchaoui/pasdebonneoudemauvaisesituation/src",
)
from pdbms.utils.snr import silero_vad_mask  # type: ignore

AMI_ROOT = Path(
    "/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice"
)
MEETINGS = ["IS1008a", "ES2011a"]
SR = 16_000

DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/"
    "vocal_helper_diar_embedding_2026-06-30.log"
)


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(DEFAULT_LOG, "a") as f:
        f.write(msg + "\n")


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


def load_speaker_spans(rttm: Path, bridge_s: float = 0.2) -> list[tuple[float, float, str]]:
    """Word-level RTTM → speaker turns with ``bridge_s`` adjacency merge."""
    rows: list[tuple[float, float, str]] = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        if len(p) < 10 or p[0] != "SPEAKER":
            continue
        t0 = float(p[3])
        rows.append((t0, t0 + float(p[4]), p[7]))
    rows.sort()
    out: list[list] = []
    for t0, t1, spk in rows:
        if out and out[-1][2] == spk and t0 - out[-1][1] <= bridge_s:
            out[-1][1] = max(out[-1][1], t1)
        else:
            out.append([t0, t1, spk])
    return [tuple(r) for r in out]


def carve_voiced_segments(
    audio: np.ndarray, vad_mask: np.ndarray, sr: int,
    min_ms: int = 500,
) -> list[tuple[float, float, np.ndarray]]:
    """Voiced runs from the Silero mask, returned with audio slices."""
    n = audio.shape[0]
    out: list[tuple[float, float, np.ndarray]] = []
    in_run = False
    run_lo = 0
    for i, v in enumerate(vad_mask):
        if v and not in_run:
            in_run = True
            run_lo = i
        elif not v and in_run:
            in_run = False
            run_hi = i
            dur_ms = (run_hi - run_lo) * 1000 / sr
            if dur_ms >= min_ms:
                out.append((
                    run_lo / sr,
                    run_hi / sr,
                    audio[run_lo:run_hi].copy(),
                ))
    if in_run:
        run_hi = n
        dur_ms = (run_hi - run_lo) * 1000 / sr
        if dur_ms >= min_ms:
            out.append((run_lo / sr, run_hi / sr, audio[run_lo:run_hi].copy()))
    return out


def label_segment(
    t0: float, t1: float,
    spans: list[tuple[float, float, str]],
) -> str | None:
    """Pick the speaker with the largest overlap with [t0, t1]."""
    overlaps: dict[str, float] = {}
    for s0, s1, spk in spans:
        lo, hi = max(t0, s0), min(t1, s1)
        if hi > lo:
            overlaps[spk] = overlaps.get(spk, 0.0) + (hi - lo)
    if not overlaps:
        return None
    return max(overlaps.items(), key=lambda kv: kv[1])[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float((a / na) @ (b / nb))


def evaluate(
    embedder, label: str, segments: list[tuple[float, float, np.ndarray, str]],
) -> dict:
    """Run all embeddings, compute intra/inter cosine distances."""
    by_spk: dict[str, list[np.ndarray]] = {}
    wall_per_call: list[float] = []
    fail = 0
    for _, _, pcm, spk in segments:
        t0 = time.perf_counter()
        try:
            emb = embedder.embed(pcm, SR)
        except Exception:  # noqa: BLE001
            fail += 1
            continue
        wall_per_call.append(time.perf_counter() - t0)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        nrm = float(np.linalg.norm(emb))
        if nrm > 0:
            emb = emb / nrm
        by_spk.setdefault(spk, []).append(emb)

    intra: list[float] = []
    inter: list[float] = []
    spk_list = list(by_spk.keys())
    for spk, embs in by_spk.items():
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                intra.append(cosine(embs[i], embs[j]))
    for i in range(len(spk_list)):
        for j in range(i + 1, len(spk_list)):
            a_embs = by_spk[spk_list[i]]
            b_embs = by_spk[spk_list[j]]
            for ea in a_embs:
                for eb in b_embs:
                    inter.append(cosine(ea, eb))

    med_intra = statistics.median(intra) if intra else float("nan")
    med_inter = statistics.median(inter) if inter else float("nan")
    margin = med_inter - med_intra
    med_wall = statistics.median(wall_per_call) if wall_per_call else float("nan")
    n_segs = sum(len(v) for v in by_spk.values())
    return {
        "label": label,
        "n_segs": n_segs,
        "n_speakers": len(by_spk),
        "fail": fail,
        "med_intra_cos": med_intra,
        "med_inter_cos": med_inter,
        "margin": margin,
        "med_wall_per_call_s": med_wall,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-token", default=None)
    args = p.parse_args()

    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")
    log(f"# Diar embedding backend sweep — 2026-06-30")
    log(f"# meetings : {MEETINGS}")

    token = resolve_hf_token(args.hf_token)

    # Build both embedders once.
    pyannote = _PyannoteEmbedder(hf_token=token)
    log("\n[setup] loading pyannote/embedding …")
    pyannote.load()
    titanet = _TitaNetEmbedder()
    log("[setup] loading TitaNet (NeMo) …")
    titanet.load()

    per_meeting: dict[str, dict[str, dict]] = {}

    for m in MEETINGS:
        mdir = AMI_ROOT / m
        wav = mdir / "mix.wav"
        rttm = mdir / "words.rttm"
        if not wav.exists() or not rttm.exists():
            log(f"\n{m} : missing files, skipping")
            continue
        audio, sr_in = read_mono_wav(wav)
        assert sr_in == SR
        dur = audio.shape[0] / SR
        spans = load_speaker_spans(rttm)
        log(f"\n=== {m} === dur={dur:.0f}s  n_ref_speakers={len(set(s[2] for s in spans))}")

        log("  running Silero VAD …")
        vad_mask = silero_vad_mask(audio, sample_rate=SR)
        voiced = carve_voiced_segments(audio, vad_mask, SR, min_ms=500)
        log(f"  voiced segments  : {len(voiced)}")

        # Label every voiced segment with the dominant ground-truth speaker.
        labelled: list[tuple[float, float, np.ndarray, str]] = []
        for t0, t1, pcm in voiced:
            spk = label_segment(t0, t1, spans)
            if spk is None:
                continue
            labelled.append((t0, t1, pcm, spk))
        log(f"  labelled segments: {len(labelled)}")

        per_meeting[m] = {}
        for embedder, name in [(pyannote, "pyannote"), (titanet, "titanet")]:
            log(f"  evaluating {name} …")
            r = evaluate(embedder, name, labelled)
            per_meeting[m][name] = r
            log(
                f"    {name:<9s} "
                f"n_segs={r['n_segs']:>3d}  fail={r['fail']:>2d}  "
                f"intra={r['med_intra_cos']:.3f}  "
                f"inter={r['med_inter_cos']:.3f}  "
                f"margin={r['margin']:>+.3f}  "
                f"wall/call={r['med_wall_per_call_s']*1000:.0f} ms"
            )

    # ----- pooled comparison -----
    log("\n" + "=" * 72)
    log("Pooled (median across meetings)")
    log("=" * 72)
    log(f"{'backend':<10s}  {'margin':>8s}  {'intra':>7s}  {'inter':>7s}  {'wall/call':>10s}")
    log("-" * 50)
    pooled: dict[str, dict[str, float]] = {}
    for backend in ["pyannote", "titanet"]:
        margins = [per_meeting[m][backend]["margin"] for m in per_meeting]
        intras = [per_meeting[m][backend]["med_intra_cos"] for m in per_meeting]
        inters = [per_meeting[m][backend]["med_inter_cos"] for m in per_meeting]
        walls = [per_meeting[m][backend]["med_wall_per_call_s"] for m in per_meeting]
        if not margins:
            continue
        pooled[backend] = {
            "margin": statistics.median(margins),
            "intra": statistics.median(intras),
            "inter": statistics.median(inters),
            "wall_per_call_ms": statistics.median(walls) * 1000,
        }
        p = pooled[backend]
        log(
            f"{backend:<10s}  "
            f"{p['margin']:>+8.3f}  "
            f"{p['intra']:>7.3f}  "
            f"{p['inter']:>7.3f}  "
            f"{p['wall_per_call_ms']:>10.0f}"
        )

    if pooled:
        winner = max(pooled.items(), key=lambda kv: kv[1]["margin"])
        log(
            f"\nWinner by separability margin : {winner[0]}  "
            f"margin={winner[1]['margin']:+.3f}"
        )

    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(json.dumps({
        "meetings": MEETINGS,
        "per_meeting": per_meeting,
        "pooled": pooled,
    }, indent=2))
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


if __name__ == "__main__":
    main()
