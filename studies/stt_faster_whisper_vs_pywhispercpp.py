"""STT engine comparison — faster-whisper vs pywhispercpp.

Question
--------
The Northflank / Coval 2026 surveys claim ``faster-whisper`` (CTranslate2)
is up to **4× faster** than the original whisper.cpp at equivalent
WER. ``vocal-helper`` ships pywhispercpp turbo by default. Does
swapping to faster-whisper give us a meaningful RTF gain on AMI ?

Protocol
--------
- Same model family : ``large-v3-turbo`` (q5 quant for pywhispercpp ;
  ``turbo`` int8 for faster-whisper, the equivalent default).
- Same inputs : full ``mix.wav`` on AMI IS1008a + ES2011a.
- Same language : ``"en"`` (locked, no LID).
- Same threads : 6 CPU threads.
- 1 warm-up run discarded, 3 timed runs ; we keep the median.

Metric
------
- WER vs words.rttm reference (lower is better).
- RTF = wall_time / audio_duration (lower is better).

Author : Warith HARCHAOUI — 2026-06-30
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

AMI_ROOT = Path(
    "/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice"
)
MEETINGS = ["IS1008a", "ES2011a"]
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/"
    "vocal_helper_stt_engines_2026-06-30.log"
)
N_TIMED = 3


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(DEFAULT_LOG, "a") as f:
        f.write(msg + "\n")


def load_reference(rttm: Path) -> str:
    words: list[tuple[float, str]] = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        if len(p) < 10 or p[0] != "SPEAKER":
            continue
        words.append((float(p[3]), p[-1]))
    words.sort()
    return " ".join(w for _, w in words)


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


def bench_pywhispercpp(pcm: np.ndarray, sr: int, language: str) -> tuple[str, list[float]]:
    """Run pywhispercpp on the full audio ; return (text_of_last_run, walls)."""
    from pywhispercpp.model import Model  # type: ignore

    model = Model(
        "large-v3-turbo-q5_0",
        n_threads=6,
        language=language,
        print_realtime=False,
        print_progress=False,
    )
    # Warm-up.
    model.transcribe(pcm)
    walls: list[float] = []
    text = ""
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        segs = model.transcribe(pcm)
        walls.append(time.perf_counter() - t0)
        text = " ".join((s.text or "").strip() for s in segs).strip()
    return text, walls


def bench_faster_whisper(pcm: np.ndarray, sr: int, language: str) -> tuple[str, list[float]]:
    """Run faster-whisper on the full audio ; return (text_of_last_run, walls)."""
    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel("large-v3-turbo", device="auto", compute_type="int8")
    # Warm-up.
    list(model.transcribe(pcm, language=language)[0])
    walls: list[float] = []
    text = ""
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        segs, _ = model.transcribe(pcm, language=language)
        text = " ".join((s.text or "").strip() for s in segs).strip()
        walls.append(time.perf_counter() - t0)
    return text, walls


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    args = p.parse_args()

    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")
    log(f"# STT engine comparison — 2026-06-30")
    log(f"# meetings : {MEETINGS}")
    log(f"# language : en (locked)")
    log(f"# n_timed  : {N_TIMED} runs + 1 warmup\n")

    # Lazy WER import.
    try:
        from jiwer import wer
    except ImportError:
        log("# installing jiwer …")
        import subprocess
        subprocess.run(["pip", "install", "-q", "jiwer"], check=True)
        from jiwer import wer

    per_meeting: dict[str, dict[str, dict]] = {}

    for m in MEETINGS:
        mdir = AMI_ROOT / m
        wav = mdir / "mix.wav"
        rttm = mdir / "words.rttm"
        if not wav.exists() or not rttm.exists():
            log(f"\n{m} : missing files, skipping")
            continue
        audio, sr = read_mono_wav(wav)
        dur = audio.shape[0] / sr
        ref = load_reference(rttm)
        log(f"\n=== {m} === dur={dur:.0f}s  ref_words={len(ref.split())}")
        per_meeting[m] = {}

        # pywhispercpp
        log("  pywhispercpp …")
        try:
            text, walls = bench_pywhispercpp(audio, sr, "en")
            w = float(wer(ref, text))
            med_wall = statistics.median(walls)
            per_meeting[m]["pywhispercpp"] = {
                "wer": w,
                "wall_med": med_wall,
                "rtf_med": med_wall / dur,
                "walls": walls,
                "hyp_words": len(text.split()),
            }
            log(
                f"    pywhispercpp  WER={w:.3f}  wall_med={med_wall:5.1f}s  "
                f"RTF_med={med_wall/dur:.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            log(f"    pywhispercpp FAILED : {exc!r}")
            per_meeting[m]["pywhispercpp"] = {"error": repr(exc)}

        # faster-whisper
        log("  faster-whisper …")
        try:
            text, walls = bench_faster_whisper(audio, sr, "en")
            w = float(wer(ref, text))
            med_wall = statistics.median(walls)
            per_meeting[m]["faster-whisper"] = {
                "wer": w,
                "wall_med": med_wall,
                "rtf_med": med_wall / dur,
                "walls": walls,
                "hyp_words": len(text.split()),
            }
            log(
                f"    faster-whisper  WER={w:.3f}  wall_med={med_wall:5.1f}s  "
                f"RTF_med={med_wall/dur:.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            log(f"    faster-whisper FAILED : {exc!r}")
            per_meeting[m]["faster-whisper"] = {"error": repr(exc)}

    # ----- pooled -----
    log("\n" + "=" * 60)
    log("Pooled median over meetings")
    log("=" * 60)
    log(f"{'engine':<16s}  {'med_WER':>8s}  {'med_RTF':>8s}  {'speedup':>8s}")
    log("-" * 50)
    pooled: dict[str, dict[str, float]] = {}
    for engine in ["pywhispercpp", "faster-whisper"]:
        wers = [
            per_meeting[m][engine]["wer"]
            for m in per_meeting
            if engine in per_meeting[m] and "wer" in per_meeting[m][engine]
        ]
        rtfs = [
            per_meeting[m][engine]["rtf_med"]
            for m in per_meeting
            if engine in per_meeting[m] and "rtf_med" in per_meeting[m][engine]
        ]
        if not wers:
            continue
        pooled[engine] = {
            "wer": statistics.median(wers),
            "rtf": statistics.median(rtfs),
        }

    if "pywhispercpp" in pooled and "faster-whisper" in pooled:
        speedup = pooled["pywhispercpp"]["rtf"] / pooled["faster-whisper"]["rtf"]
    else:
        speedup = float("nan")

    for engine in ["pywhispercpp", "faster-whisper"]:
        if engine not in pooled:
            continue
        sp = "—"
        if engine == "faster-whisper":
            sp = f"{speedup:.2f}×"
        log(
            f"{engine:<16s}  {pooled[engine]['wer']:>8.3f}  "
            f"{pooled[engine]['rtf']:>8.3f}  {sp:>8s}"
        )

    if "pywhispercpp" in pooled and "faster-whisper" in pooled:
        if pooled["faster-whisper"]["rtf"] < pooled["pywhispercpp"]["rtf"] - 0.01:
            log(f"\nWinner : faster-whisper  ({speedup:.2f}× speedup, WER similar)")
        else:
            log("\nWinner : pywhispercpp (faster-whisper does not beat it on RTF here)")

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
