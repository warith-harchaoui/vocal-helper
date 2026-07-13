"""Picovoice Falcon vs pyannote.audio — DER and compute.

Question
--------
Picovoice's Falcon Speaker Diarization claims (December 2025 update,
https://picovoice.ai/blog/state-of-speaker-diarization/) :

- DER : 10.3 % (Falcon) vs 9.0 % (pyannote) on the published benchmark
  → pyannote +1.3 pp DER lead
- JER : 19.9 % (Falcon) vs 27.4 % (pyannote)
  → Falcon −7.5 pp JER advantage
- Compute : 221 × less core-hours per 100 h audio
- Memory : 0.1 GiB (Falcon) vs 1.5 GiB (pyannote) — 15 × footprint

For our **offline** path, the compute / memory advantage is huge if
the quality gap holds on AMI.

Protocol
--------
- corpus : AMI dev-slice (IS1008a + ES2011a)
- backend Falcon : ``pvfalcon`` Python package (requires
  ``PICOVOICE_ACCESS_KEY`` env var ; free tier ≥ 10 h / month is
  enough for this study).
- backend pyannote : ``pyannote.audio.Pipeline("pyannote/speaker-diarization-3.1")``
  — same path :class:`vocal_helper.diar.OfflineDiarStage` uses.
- DER computed via ``pyannote.metrics.diarization.DiarizationErrorRate``
  with ``collar=0.25`` and ``skip_overlap=False`` — same canonical
  protocol as the 2026-06-30 stitch-threshold sweep.

If ``pvfalcon`` is not installed or the access key is missing, the
study logs a clear message and skips Falcon (we don't fail the whole
cascade).

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

AMI_ROOT = Path(
    "/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice"
)
MEETINGS = ["IS1008a", "ES2011a"]
SR = 16_000
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/"
    "vocal_helper_diar_falcon_2026-06-30.log"
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


def load_reference_annotation(rttm: Path):
    """Word-level RTTM → pyannote Annotation, 200 ms speaker bridging."""
    from pyannote.core import Annotation, Segment

    rows: list[tuple[float, float, str]] = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        if len(p) < 10 or p[0] != "SPEAKER":
            continue
        t0 = float(p[3])
        rows.append((t0, t0 + float(p[4]), p[7]))
    rows.sort()
    by: dict[str, list[list[float]]] = {}
    for t0, t1, spk in rows:
        if t1 <= t0:
            continue
        b = by.setdefault(spk, [])
        if b and t0 - b[-1][1] <= 0.20:
            b[-1][1] = max(b[-1][1], t1)
        else:
            b.append([t0, t1])
    ann = Annotation()
    for spk, turns in by.items():
        for t0, t1 in turns:
            ann[Segment(t0, t1)] = spk
    return ann


def hypothesis_to_annotation(segs: list[tuple[float, float, str]]):
    """``[(t0, t1, speaker), …]`` → pyannote Annotation."""
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for t0, t1, spk in segs:
        if t1 <= t0:
            continue
        ann[Segment(t0, t1)] = str(spk)
    return ann


# ---------------------------------------------------------------------------
# Backend wrappers.
# ---------------------------------------------------------------------------


def run_falcon(audio: np.ndarray, sr: int) -> tuple[list[tuple[float, float, str]], float]:
    """Pico Falcon ; returns (segments, wall_seconds)."""
    import tempfile
    import pvfalcon  # type: ignore

    access_key = os.environ.get("PICOVOICE_ACCESS_KEY", "")
    if not access_key:
        raise RuntimeError(
            "PICOVOICE_ACCESS_KEY is not set in the environment. "
            "Sign up at https://console.picovoice.ai/ (free tier) and "
            "export the key."
        )

    falcon = pvfalcon.create(access_key=access_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, sr, subtype="PCM_16")
        t0 = time.perf_counter()
        segs = falcon.process_file(tmp.name)
        wall = time.perf_counter() - t0
    falcon.delete()
    # ``segs`` is a list of namedtuples with ``start_sec``, ``end_sec``, ``speaker_tag``.
    out = [(s.start_sec, s.end_sec, f"S{s.speaker_tag}") for s in segs]
    return out, wall


def run_pyannote(audio: np.ndarray, sr: int, hf_token: str | None) -> tuple[list[tuple[float, float, str]], float]:
    """pyannote/speaker-diarization-3.1 ; returns (segments, wall_seconds)."""
    import torch
    from pyannote.audio import Pipeline

    # pyannote.audio renamed the HF auth kwarg between major versions
    # — try the new name first and fall back so the bench works
    # against both 3.x and 4.x.
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=hf_token,
        )
    except TypeError:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token,
        )
    wave = torch.from_numpy(audio).unsqueeze(0)
    t0 = time.perf_counter()
    ann = pipe({"waveform": wave, "sample_rate": sr})
    wall = time.perf_counter() - t0
    out = [
        (seg.start, seg.end, str(spk))
        for seg, _track, spk in ann.itertracks(yield_label=True)
    ]
    return out, wall


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--meetings", nargs="*", default=MEETINGS)
    args = p.parse_args()

    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")
    log("# Falcon vs pyannote — 2026-06-30")
    log(f"# meetings : {args.meetings}")

    if not args.hf_token:
        # Read settings.yaml as a last resort.
        try:
            from vocal_helper._settings import resolve_hf_token
            args.hf_token = resolve_hf_token(None)
        except Exception:
            pass
    log(f"# hf_token  : {'<set>' if args.hf_token else '<missing>'}")
    log(f"# pv_key    : {'<set>' if os.environ.get('PICOVOICE_ACCESS_KEY') else '<missing>'}")

    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError as exc:
        log(f"missing pyannote.metrics : {exc} — abort")
        return

    per_meeting: dict[str, dict[str, dict]] = {}

    for m in args.meetings:
        mdir = AMI_ROOT / m
        wav = mdir / "mix.wav"
        rttm = mdir / "words.rttm"
        if not wav.exists() or not rttm.exists():
            log(f"\n{m} : missing files, skipping")
            continue
        audio, sr_in = read_mono_wav(wav)
        assert sr_in == SR
        dur = audio.shape[0] / SR
        ref_ann = load_reference_annotation(rttm)
        n_ref_spk = len(set(ref_ann.labels()))
        log(f"\n=== {m} === dur={dur:.0f}s  n_ref_speakers={n_ref_spk}")

        per_meeting[m] = {}

        # ---- Falcon ----
        try:
            log("  Falcon …")
            segs, wall = run_falcon(audio, SR)
            metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
            der = float(metric(ref_ann, hypothesis_to_annotation(segs)))
            rtf = wall / dur
            per_meeting[m]["falcon"] = {
                "der": der, "wall": wall, "rtf": rtf, "n_segs": len(segs),
            }
            log(f"    falcon   DER={der:.3f}  RTF={rtf:.3f}  wall={wall:.1f}s  n_segs={len(segs)}")
        except Exception as exc:  # noqa: BLE001
            log(f"    falcon FAILED : {exc!r}")
            per_meeting[m]["falcon"] = {"error": repr(exc)}

        # ---- pyannote ----
        try:
            log("  pyannote 3.1 …")
            segs, wall = run_pyannote(audio, SR, args.hf_token)
            metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
            der = float(metric(ref_ann, hypothesis_to_annotation(segs)))
            rtf = wall / dur
            per_meeting[m]["pyannote"] = {
                "der": der, "wall": wall, "rtf": rtf, "n_segs": len(segs),
            }
            log(f"    pyannote DER={der:.3f}  RTF={rtf:.3f}  wall={wall:.1f}s  n_segs={len(segs)}")
        except Exception as exc:  # noqa: BLE001
            log(f"    pyannote FAILED : {exc!r}")
            per_meeting[m]["pyannote"] = {"error": repr(exc)}

    # ----- pooled summary -----
    log("\n" + "=" * 56)
    log("Pooled median over meetings")
    log("=" * 56)
    log(f"{'backend':<10s}  {'med_DER':>8s}  {'med_RTF':>8s}")
    log("-" * 34)
    pooled: dict[str, dict[str, float]] = {}
    for backend in ["falcon", "pyannote"]:
        ders = [
            per_meeting[m][backend]["der"]
            for m in per_meeting
            if backend in per_meeting[m] and "der" in per_meeting[m][backend]
        ]
        rtfs = [
            per_meeting[m][backend]["rtf"]
            for m in per_meeting
            if backend in per_meeting[m] and "rtf" in per_meeting[m][backend]
        ]
        if not ders:
            continue
        pooled[backend] = {
            "der": statistics.median(ders),
            "rtf": statistics.median(rtfs),
        }
        log(f"{backend:<10s}  {pooled[backend]['der']:>8.3f}  {pooled[backend]['rtf']:>8.3f}")

    if "falcon" in pooled and "pyannote" in pooled:
        d_gap = pooled["falcon"]["der"] - pooled["pyannote"]["der"]
        r_speedup = pooled["pyannote"]["rtf"] / pooled["falcon"]["rtf"]
        log(f"\nDER gap (falcon − pyannote) : {d_gap:+.3f}")
        log(f"RTF speedup (falcon vs pyannote) : {r_speedup:.1f}×")
        # Decision rule : if Falcon's DER is within 0.03 of pyannote
        # AND it's ≥ 5× faster, recommend Falcon for the offline
        # path — the compute saving dominates the small quality loss.
        if d_gap <= 0.03 and r_speedup >= 5:
            log("\nRecommendation : adopt Falcon for OfflineDiarStage default — "
                "compute win dominates.")
        elif d_gap > 0.03:
            log("\nRecommendation : keep pyannote — Falcon's DER gap is too large here.")
        else:
            log("\nRecommendation : keep pyannote — RTF speedup is marginal.")

    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(json.dumps({
        "meetings": args.meetings,
        "per_meeting": per_meeting,
        "pooled": pooled,
    }, indent=2))
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


if __name__ == "__main__":
    main()
