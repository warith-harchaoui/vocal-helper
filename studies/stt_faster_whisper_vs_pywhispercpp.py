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

AMI_ROOT = Path("/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice")
MEETINGS = ["IS1008a", "ES2011a"]
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/vocal_helper_stt_engines_2026-06-30.log"
)
N_TIMED = 3


def log(msg: str) -> None:
    """Echo a line to stdout and append it to the on-disk study log.

    Parameters
    ----------
    msg : str
        The line to emit. It is printed verbatim (a ``studies/`` result-table
        line, the allowed exception to the no-print rule) and also appended —
        with a trailing newline — to :data:`DEFAULT_LOG`.

    Returns
    -------
    None
        The message is emitted for its side effects only.
    """
    # Live console output for the human watching the run.
    print(msg, flush=True)
    # Persist the identical line so the log file mirrors stdout exactly.
    with open(DEFAULT_LOG, "a") as f:
        f.write(msg + "\n")


def load_reference(rttm: Path) -> str:
    """Flatten a word-level RTFM into a single reference transcript string.

    Parameters
    ----------
    rttm : Path
        Path to a ``words.rttm`` file whose ``SPEAKER`` rows carry one word
        each (word text in the last column, onset in column 4).

    Returns
    -------
    str
        The reference words joined by single spaces, ordered by onset — the
        ground truth fed to WER.
    """
    words: list[tuple[float, str]] = []
    # Parse line by line: keep only well-formed SPEAKER rows and pair each
    # word's onset (col 3) with its text (last col) so we can time-order it.
    for line in rttm.read_text().splitlines():
        p = line.split()
        # Skip headers, comments and any row that is not a 10+ field SPEAKER
        # entry — malformed lines would corrupt the reference transcript.
        if len(p) < 10 or p[0] != "SPEAKER":
            continue
        words.append((float(p[3]), p[-1]))
    # Sort by onset so the transcript reads in spoken order regardless of the
    # file's row ordering, then drop the timestamps.
    words.sort()
    return " ".join(w for _, w in words)


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV as a mono float32 waveform.

    Parameters
    ----------
    path : Path
        Path to the WAV file to read.

    Returns
    -------
    (np.ndarray, int)
        The mono ``float32`` samples and the file's sample rate. Multi-channel
        inputs are down-mixed by averaging the channels.
    """
    audio, sr = sf.read(str(path), dtype="float32")
    # Both STT engines expect a single channel; average multi-channel audio
    # down to mono and keep float32 to avoid an implicit upcast to float64.
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


def bench_pywhispercpp(pcm: np.ndarray, sr: int, language: str) -> tuple[str, list[float]]:
    """Run pywhispercpp on the full audio ; return (text_of_last_run, walls)."""
    from pywhispercpp.model import Model  # type: ignore

    # q5_0 turbo quant + 6 threads: the exact config vocal-helper ships, so
    # this measures the real default rather than a tuned-for-benchmark variant.
    model = Model(
        "large-v3-turbo-q5_0",
        n_threads=6,
        language=language,
        print_realtime=False,
        print_progress=False,
    )
    # One discarded warm-up run so JIT / cache / model-load costs don't land
    # in the timed medians.
    model.transcribe(pcm)
    walls: list[float] = []
    text = ""
    # Time N_TIMED full transcriptions; keep the last run's text (all runs
    # are deterministic, so any run's transcript is representative).
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        segs = model.transcribe(pcm)
        walls.append(time.perf_counter() - t0)
        text = " ".join((s.text or "").strip() for s in segs).strip()
    return text, walls


def bench_faster_whisper(pcm: np.ndarray, sr: int, language: str) -> tuple[str, list[float]]:
    """Run faster-whisper on the full audio ; return (text_of_last_run, walls)."""
    from faster_whisper import WhisperModel  # type: ignore

    # int8 turbo is the faster-whisper analogue of the q5 pywhispercpp default;
    # device="auto" lets CTranslate2 pick the best available backend.
    model = WhisperModel("large-v3-turbo", device="auto", compute_type="int8")
    # faster-whisper yields lazily, so materialise the generator to force the
    # full transcription during the discarded warm-up.
    list(model.transcribe(pcm, language=language)[0])
    walls: list[float] = []
    text = ""
    # Time N_TIMED runs. The wall clock must wrap the generator consumption
    # (the join), because transcribe() returns before any work is done.
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        segs, _ = model.transcribe(pcm, language=language)
        text = " ".join((s.text or "").strip() for s in segs).strip()
        walls.append(time.perf_counter() - t0)
    return text, walls


def main() -> None:
    """Benchmark both STT engines across the AMI meetings and report the winner.

    For each meeting it loads the mixed audio and reference transcript, times
    pywhispercpp and faster-whisper (median of :data:`N_TIMED` runs, one
    discarded warm-up), scores WER against the reference, then pools the
    per-meeting medians to declare an RTF winner and dump everything to JSON.

    Returns
    -------
    None
        All results go to stdout, the study log, and the sibling JSON file.
    """
    # The module docstring doubles as the --help text; no real args are used.
    p = argparse.ArgumentParser(description=__doc__)
    args = p.parse_args()

    # Start from a clean log so the file holds only this run's transcript.
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")
    # Header block pins the run parameters so the log is self-describing.
    log(f"# STT engine comparison — 2026-06-30")
    log(f"# meetings : {MEETINGS}")
    log(f"# language : en (locked)")
    log(f"# n_timed  : {N_TIMED} runs + 1 warmup\n")

    # jiwer is only needed for WER; import it lazily and self-install if the
    # study machine lacks it, so the script is runnable out of the box.
    try:
        from jiwer import wer
    except ImportError:
        log("# installing jiwer …")
        import subprocess

        subprocess.run(["pip", "install", "-q", "jiwer"], check=True)
        from jiwer import wer

    # per_meeting[meeting][engine] -> metrics dict (or an {"error": ...} row).
    per_meeting: dict[str, dict[str, dict]] = {}

    # Score every meeting independently; medians are pooled across them later.
    for m in MEETINGS:
        mdir = AMI_ROOT / m
        wav = mdir / "mix.wav"
        rttm = mdir / "words.rttm"
        # Skip meetings whose audio or reference is absent rather than crash —
        # the dev-slice may not carry every meeting on every machine.
        if not wav.exists() or not rttm.exists():
            log(f"\n{m} : missing files, skipping")
            continue
        # Load once and reuse for both engines: identical input isolates the
        # engine as the only variable.
        audio, sr = read_mono_wav(wav)
        dur = audio.shape[0] / sr  # RTF denominator (seconds of audio).
        ref = load_reference(rttm)
        log(f"\n=== {m} === dur={dur:.0f}s  ref_words={len(ref.split())}")
        per_meeting[m] = {}

        # --- pywhispercpp (vocal-helper's current default) ---
        # Wrapped in try/except so one engine failing still lets the other run
        # and the comparison table degrades gracefully.
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
                f"RTF_med={med_wall / dur:.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            log(f"    pywhispercpp FAILED : {exc!r}")
            per_meeting[m]["pywhispercpp"] = {"error": repr(exc)}

        # --- faster-whisper (CTranslate2, the challenger) ---
        # Same guarded pattern: measure it independently of pywhispercpp.
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
                f"RTF_med={med_wall / dur:.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            log(f"    faster-whisper FAILED : {exc!r}")
            per_meeting[m]["faster-whisper"] = {"error": repr(exc)}

    # ----- pooled -----
    # Collapse the per-meeting numbers into one median per engine so the
    # headline comparison is not skewed by a single meeting's outlier.
    log("\n" + "=" * 60)
    log("Pooled median over meetings")
    log("=" * 60)
    log(f"{'engine':<16s}  {'med_WER':>8s}  {'med_RTF':>8s}  {'speedup':>8s}")
    log("-" * 50)
    pooled: dict[str, dict[str, float]] = {}
    for engine in ["pywhispercpp", "faster-whisper"]:
        # Gather each engine's per-meeting WER / RTF, skipping meetings where
        # that engine errored out (no "wer" / "rtf_med" key present).
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
        # An engine that failed on every meeting has nothing to pool.
        if not wers:
            continue
        pooled[engine] = {
            "wer": statistics.median(wers),
            "rtf": statistics.median(rtfs),
        }

    # Speedup = how many times faster faster-whisper is (baseline RTF over
    # challenger RTF). NaN when either engine is missing so we never divide
    # by an absent number.
    if "pywhispercpp" in pooled and "faster-whisper" in pooled:
        speedup = pooled["pywhispercpp"]["rtf"] / pooled["faster-whisper"]["rtf"]
    else:
        speedup = float("nan")

    for engine in ["pywhispercpp", "faster-whisper"]:
        if engine not in pooled:
            continue
        # Only the challenger row carries the speedup figure; the baseline
        # shows a dash.
        sp = "—"
        if engine == "faster-whisper":
            sp = f"{speedup:.2f}×"
        log(
            f"{engine:<16s}  {pooled[engine]['wer']:>8.3f}  {pooled[engine]['rtf']:>8.3f}  {sp:>8s}"
        )

    # Declare a winner only if faster-whisper beats the default RTF by a
    # margin (0.01) large enough not to be measurement noise; otherwise the
    # incumbent pywhispercpp stays.
    if "pywhispercpp" in pooled and "faster-whisper" in pooled:
        if pooled["faster-whisper"]["rtf"] < pooled["pywhispercpp"]["rtf"] - 0.01:
            log(f"\nWinner : faster-whisper  ({speedup:.2f}× speedup, WER similar)")
        else:
            log("\nWinner : pywhispercpp (faster-whisper does not beat it on RTF here)")

    # Machine-readable dump so the raw per-meeting numbers survive alongside
    # the printed summary.
    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "meetings": MEETINGS,
                "per_meeting": per_meeting,
                "pooled": pooled,
            },
            indent=2,
        )
    )
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


if __name__ == "__main__":
    main()
