#!/usr/bin/env python3
"""Offline ASR — per-segment vs concatenated big-chunk batching sweep.

Goal
----
The offline pipeline (:class:`vocal_helper.pipeline.OfflinePipeline`)
diarizes the whole buffer, then transcribes it **one diarized segment
at a time** (:class:`vocal_helper.asr.WhisperStage` awaits each
``to_thread`` transcribe before the next). whisper.cpp pads every mel
spectrogram to a fixed 30 s window, so a 0.8 s turn costs roughly the
same encoder pass as a 25 s turn. A meeting with hundreds of short
turns therefore pays hundreds of ~30 s encodes.

This study measures the "full-throttle" alternative : greedily
**concatenate consecutive diarized segments** into chunks capped at
``max_chunk_s`` and run **one whisper call per chunk**. Fewer, fuller
calls ⇒ the fixed 30 s encoder cost is amortised. The open question is
whether the longer context *helps* WER (more decoder context) or
*hurts* it (a chunk spanning an en→fr speaker switch forces whisper to
pick one language).

Method
------
1. Reference clips : ``data/bagarre-rich/mix_*.wav`` — 30 s bilingual
   (en/fr) multi-speaker mixes with a word-level ``.rttm`` giving each
   word's ``start / dur / speaker-gid / text``. Ground-truth diar is
   read from the RTTM so this study isolates the **ASR batching
   variable** from diarization error (same input segments feed every
   strategy).
2. Build GT "turns" : sort words by start, cut a new turn whenever the
   speaker gid changes. Each turn → ``(t0, t1, pcm-slice)``.
3. Strategies (one warm whisper model, reused across all runs) :
     - ``per_segment``  — transcribe each turn slice individually
                          (mirrors today's WhisperStage). n_calls = n_turns.
     - ``batch_<S>s``   — pack consecutive turns into ≤ S-second chunks
                          (0.1 s silence pad between turns), one call per
                          chunk. n_calls = n_chunks.
4. Metrics per clip : WER vs the time-ordered word reference (jiwer),
   RTF = whisper wall-time / clip-duration, and n_whisper_calls.
   Pooled as medians across clips.

The winner (lowest RTF at non-regressing WER) sets the offline
default in :class:`OfflinePipeline`.

Data root
---------
Reference audio is NOT shipped with the open-source package. Point the
script at a local corpus with ``--data-root`` (or ``VH_STUDY_DATA_ROOT``)
containing ``bagarre-rich/mix_*.wav`` + matching ``.rttm``.

Author : Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_MODEL = "large-v3-turbo-q5_0"
DEFAULT_THREADS = 6
# whisper.cpp pads the mel to 30 s ; keep chunks just under so a single
# turn never overflows a window (which would cost two encodes anyway).
DEFAULT_CHUNK_SIZES_S = (12.0, 24.0)
PAD_S = 0.1  # silence inserted between concatenated turns
_LOG = Path(__file__).with_name("asr_offline_batching.log")


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(_LOG, "a") as f:
        f.write(msg + "\n")


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


@dataclass
class Turn:
    t0: float
    t1: float
    speaker: str


def parse_rttm_words(rttm: Path) -> list[tuple[float, float, str, str]]:
    """Return ``[(start_s, end_s, speaker_gid, word)]`` sorted by start.

    RTTM word row layout (bagarre-rich) ::

        SPEAKER <id> <chan> <start> <dur> <NA> <NA> <gid> <NA> <word>
    """
    rows: list[tuple[float, float, str, str]] = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        if len(p) < 10 or p[0] != "SPEAKER":
            continue
        start = float(p[3])
        dur = float(p[4])
        gid = p[7]
        word = p[-1]
        rows.append((start, start + dur, gid, word))
    rows.sort(key=lambda r: r[0])
    return rows


def turns_from_words(words: list[tuple[float, float, str, str]]) -> list[Turn]:
    """Cut a new turn whenever the speaker gid changes (time order)."""
    turns: list[Turn] = []
    for start, end, gid, _word in words:
        if turns and turns[-1].speaker == gid:
            turns[-1].t1 = max(turns[-1].t1, end)
        else:
            turns.append(Turn(t0=start, t1=end, speaker=gid))
    return turns


def reference_text(words: list[tuple[float, float, str, str]]) -> str:
    return " ".join(w for _, _, _, w in words)


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


def slice_pcm(audio: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    a = max(0, int(round(t0 * sr)))
    b = min(audio.shape[0], int(round(t1 * sr)))
    return audio[a:b]


def pack_chunks(
    turns: list[Turn], max_chunk_s: float, *, speaker_coherent: bool = False
) -> list[list[Turn]]:
    """Greedily group consecutive turns into ≤ max_chunk_s windows.

    ``speaker_coherent=True`` also breaks a chunk on every speaker
    change, so each chunk holds exactly one speaker. That keeps
    word→speaker attribution trivial (the whole chunk is one speaker)
    and avoids forcing whisper to pick one language across an
    en→fr switch — at the cost of more, smaller chunks.
    """
    chunks: list[list[Turn]] = []
    cur: list[Turn] = []
    cur_dur = 0.0
    for t in turns:
        d = t.t1 - t.t0
        breaks = cur and cur_dur + PAD_S + d > max_chunk_s
        if speaker_coherent and cur and cur[-1].speaker != t.speaker:
            breaks = True
        if breaks:
            chunks.append(cur)
            cur, cur_dur = [], 0.0
        cur.append(t)
        cur_dur += (PAD_S if cur_dur else 0.0) + d
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def transcribe(model, pcm: np.ndarray) -> str:
    if pcm.shape[0] < int(0.05 * 16000):  # < 50 ms — whisper hallucinates
        return ""
    segs = model.transcribe(pcm)
    return " ".join((s.text or "").strip() for s in segs).strip()


def run_per_segment(model, audio, sr, turns) -> tuple[str, float, int]:
    parts: list[str] = []
    t0 = time.perf_counter()
    for t in turns:
        parts.append(transcribe(model, slice_pcm(audio, sr, t.t0, t.t1)))
    wall = time.perf_counter() - t0
    return " ".join(p for p in parts if p).strip(), wall, len(turns)


def run_batched(model, audio, sr, turns, max_chunk_s, *, speaker_coherent=False) -> tuple[str, float, int]:
    chunks = pack_chunks(turns, max_chunk_s, speaker_coherent=speaker_coherent)
    pad = np.zeros(int(PAD_S * sr), dtype=np.float32)
    parts: list[str] = []
    t0 = time.perf_counter()
    for chunk in chunks:
        pieces: list[np.ndarray] = []
        for i, t in enumerate(chunk):
            if i:
                pieces.append(pad)
            pieces.append(slice_pcm(audio, sr, t.t0, t.t1))
        parts.append(transcribe(model, np.concatenate(pieces)))
    wall = time.perf_counter() - t0
    return " ".join(p for p in parts if p).strip(), wall, len(chunks)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        default=os.environ.get("VH_STUDY_DATA_ROOT", ""),
        help="dir containing bagarre-rich/mix_*.wav + .rttm",
    )
    ap.add_argument("--n", type=int, default=12, help="clips to sample")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument(
        "--chunk-sizes",
        default=",".join(str(s) for s in DEFAULT_CHUNK_SIZES_S),
        help="comma-separated max_chunk_s values",
    )
    args = ap.parse_args()

    if not args.data_root:
        raise SystemExit(
            "no --data-root ; pass a dir with bagarre-rich/ (or set VH_STUDY_DATA_ROOT)"
        )
    rich = Path(args.data_root).expanduser() / "bagarre-rich"
    wavs = sorted(rich.glob("mix_*.wav"))[: args.n]
    if not wavs:
        raise SystemExit(f"no mix_*.wav under {rich}")
    chunk_sizes = [float(x) for x in args.chunk_sizes.split(",")]

    _LOG.write_text("")
    log("# Offline ASR batching — per-segment vs concatenated chunks")
    log(f"# model={args.model} threads={args.threads} clips={len(wavs)} chunks={chunk_sizes}")

    # One warm model, reused across every strategy/clip (measures inference,
    # not load) — exactly how WhisperStage holds a single lazy instance.
    from pywhispercpp.model import Model  # type: ignore

    model = Model(args.model, n_threads=args.threads, print_realtime=False, print_progress=False)

    try:
        from jiwer import wer
    except ImportError:
        import subprocess

        log("# installing jiwer …")
        subprocess.run(["pip", "install", "-q", "jiwer"], check=True)
        from jiwer import wer

    strategies = ["per_segment"]
    for s in chunk_sizes:
        strategies.append(f"batch_{s:g}s")
        strategies.append(f"batch_spk_{s:g}s")
    rows: dict[str, list[tuple[float, float, int]]] = {s: [] for s in strategies}

    for wav in wavs:
        rttm = wav.with_suffix(".rttm")
        if not rttm.exists():
            log(f"{wav.name} : no rttm, skip")
            continue
        audio, sr = read_mono(wav)
        dur = audio.shape[0] / sr
        words = parse_rttm_words(rttm)
        turns = turns_from_words(words)
        ref = reference_text(words)
        log(f"\n{wav.name}  dur={dur:.0f}s  turns={len(turns)}  ref_words={len(ref.split())}")

        hyp, wall, calls = run_per_segment(model, audio, sr, turns)
        w = float(wer(ref, hyp)) if ref and hyp else 1.0
        rows["per_segment"].append((w, wall / dur, calls))
        log(f"  {'per_segment':<12s}  WER={w:.3f}  RTF={wall/dur:.3f}  calls={calls}")

        for s in chunk_sizes:
            for coherent, tag in ((False, ""), (True, "_spk")):
                hyp, wall, calls = run_batched(
                    model, audio, sr, turns, s, speaker_coherent=coherent
                )
                w = float(wer(ref, hyp)) if ref and hyp else 1.0
                key = f"batch{tag}_{s:g}s"
                rows[key].append((w, wall / dur, calls))
                log(f"  {key:<14s}  WER={w:.3f}  RTF={wall/dur:.3f}  calls={calls}")

    # ----- pooled medians -----
    log("\n" + "=" * 60)
    log(f"{'strategy':<14s}  {'med_WER':>8s}  {'med_RTF':>8s}  {'med_calls':>9s}")
    log("-" * 60)
    pooled: dict[str, tuple[float, float, float]] = {}
    for s in strategies:
        vals = rows[s]
        if not vals:
            continue
        mw = statistics.median(v[0] for v in vals)
        mr = statistics.median(v[1] for v in vals)
        mc = statistics.median(v[2] for v in vals)
        pooled[s] = (mw, mr, mc)
        log(f"{s:<14s}  {mw:>8.3f}  {mr:>8.3f}  {mc:>9.0f}")

    if "per_segment" in pooled:
        base_wer, base_rtf, _ = pooled["per_segment"]
        log("\nvs per_segment baseline :")
        for s, (mw, mr, _) in pooled.items():
            if s == "per_segment":
                continue
            speedup = base_rtf / mr if mr else float("inf")
            log(
                f"  {s:<12s}  RTF {speedup:.1f}× faster  "
                f"WER {mw - base_wer:+.3f} ({'better' if mw <= base_wer else 'worse'})"
            )

    _LOG.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": args.model,
                "threads": args.threads,
                "clips": [w.name for w in wavs],
                "chunk_sizes_s": chunk_sizes,
                "pooled": {k: list(v) for k, v in pooled.items()},
                "per_clip": {k: [list(v) for v in rows[k]] for k in rows},
            },
            indent=2,
        )
    )
    log(f"\nwrote {_LOG.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
