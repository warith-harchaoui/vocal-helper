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
    """Echo one line to stdout and append it to the study log file.

    Parameters
    ----------
    msg : str
        The line to emit. Printed verbatim (the study's result tables
        are stdout output — see this module's ``print`` policy) and
        mirrored to :data:`_LOG` so a headless run keeps a transcript.

    Returns
    -------
    None
    """
    # stdout carries the live table for the operator watching the run.
    print(msg, flush=True)
    # Append (never truncate) so every log() call adds to the transcript
    # started by main()'s _LOG.write_text("") reset.
    with open(_LOG, "a") as f:
        f.write(msg + "\n")


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


@dataclass
class Turn:
    """One contiguous same-speaker span of the reference timeline.

    A turn is the unit both batching strategies operate on: the
    per-segment strategy transcribes one turn per whisper call, the
    batched strategy packs several turns into a single call.

    Attributes
    ----------
    t0 : float
        Turn start, in seconds from the clip origin.
    t1 : float
        Turn end, in seconds. Extended as later words of the same
        speaker are merged in (see :func:`turns_from_words`).
    speaker : str
        Ground-truth speaker gid; a change in this value is what cuts
        one turn from the next.
    """

    t0: float
    t1: float
    speaker: str


def parse_rttm_words(rttm: Path) -> list[tuple[float, float, str, str]]:
    """Return ``[(start_s, end_s, speaker_gid, word)]`` sorted by start.

    RTTM word row layout (bagarre-rich) ::

        SPEAKER <id> <chan> <start> <dur> <NA> <NA> <gid> <NA> <word>

    Parameters
    ----------
    rttm : Path
        Word-level RTTM file paired with the mix clip.

    Returns
    -------
    list of tuple of (float, float, str, str)
        One ``(start_s, end_s, speaker_gid, word)`` per RTTM word row,
        sorted ascending by start time so downstream turn-cutting can
        walk the words in playback order.
    """
    rows: list[tuple[float, float, str, str]] = []
    # Parse each RTTM line into a word row; tolerate stray / malformed
    # lines by skipping anything that is not a full SPEAKER record.
    for line in rttm.read_text().splitlines():
        p = line.split()
        # Guard on both the tag and the column count so p[7]/p[-1] below
        # never index past a short line.
        if len(p) < 10 or p[0] != "SPEAKER":
            continue
        # RTTM stores start + duration; we carry (start, end) instead so
        # slicing and turn merging only ever deal in absolute times.
        start = float(p[3])
        dur = float(p[4])
        gid = p[7]  # global speaker id — the turn-boundary signal
        word = p[-1]  # the transcribed token sits in the last column
        rows.append((start, start + dur, gid, word))
    # Words may be interleaved across speakers in the file; sort by start
    # so turn-cutting sees a single monotonic timeline.
    rows.sort(key=lambda r: r[0])
    return rows


def turns_from_words(words: list[tuple[float, float, str, str]]) -> list[Turn]:
    """Merge a time-ordered word stream into same-speaker turns.

    Parameters
    ----------
    words : list of tuple of (float, float, str, str)
        ``(start_s, end_s, speaker_gid, word)`` rows in start order, as
        returned by :func:`parse_rttm_words`.

    Returns
    -------
    list of Turn
        Consecutive words sharing a speaker gid are folded into one
        :class:`Turn`; a gid change starts a new turn. This is the
        ground-truth segmentation both batching strategies consume.
    """
    turns: list[Turn] = []
    # Sweep words once, growing the current turn while the speaker holds
    # and opening a new one the moment the gid changes.
    for start, end, gid, _word in words:
        # Same speaker as the open turn: just push its end forward.
        # max() guards against any out-of-order word end within a turn.
        if turns and turns[-1].speaker == gid:
            turns[-1].t1 = max(turns[-1].t1, end)
        # Speaker changed (or first word): begin a fresh turn.
        else:
            turns.append(Turn(t0=start, t1=end, speaker=gid))
    return turns


def reference_text(words: list[tuple[float, float, str, str]]) -> str:
    """Flatten the word rows into a single space-joined reference string.

    Parameters
    ----------
    words : list of tuple of (float, float, str, str)
        ``(start_s, end_s, speaker_gid, word)`` rows in time order.

    Returns
    -------
    str
        All words joined by single spaces — the time-ordered reference
        transcript every strategy's WER is scored against.
    """
    # Drop the timing/speaker columns; WER only needs the token sequence.
    return " ".join(w for _, _, _, w in words)


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as a mono float32 signal.

    Parameters
    ----------
    path : Path
        Path to the mix clip on disk.

    Returns
    -------
    tuple of (numpy.ndarray, int)
        The mono ``float32`` samples and their native sample rate. No
        resampling is done here — the reference clips are already at the
        rate whisper expects.
    """
    audio, sr = sf.read(str(path), dtype="float32")
    # Downmix any stereo/multichannel clip to mono so the whole study
    # works on a single 1-D signal (whisper takes mono anyway).
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


def slice_pcm(audio: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    """Cut the ``[t0, t1)`` second window out of a PCM signal.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono PCM samples for the whole clip.
    sr : int
        Sample rate of ``audio`` in Hz.
    t0, t1 : float
        Window bounds in seconds.

    Returns
    -------
    numpy.ndarray
        The samples in the window, clamped to the signal's extent so
        rounding at the edges can never index out of bounds.
    """
    # Convert seconds to sample indices, rounding to the nearest sample,
    # and clamp both ends inside the array so edge turns stay in range.
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

    Parameters
    ----------
    turns : list of Turn
        Ground-truth turns in time order.
    max_chunk_s : float
        Upper bound on a chunk's packed duration (turn audio plus the
        inter-turn :data:`PAD_S` pads), in seconds.
    speaker_coherent : bool, optional
        When ``True``, also break on every speaker change so each chunk
        holds a single speaker (default ``False``).

    Returns
    -------
    list of list of Turn
        Consecutive-turn groups; feeding one group to whisper is one
        call. Fewer, fuller groups amortise whisper's fixed 30 s encode.
    """
    chunks: list[list[Turn]] = []
    cur: list[Turn] = []  # turns accumulated into the chunk being built
    cur_dur = 0.0  # packed duration of `cur` so far (turns + pads)
    # Greedy single pass: keep appending turns to the open chunk until a
    # break condition fires, then flush and start fresh.
    for t in turns:
        d = t.t1 - t.t0
        # Break when adding this turn (plus its leading pad) would push
        # the chunk over the cap — but only if the chunk is non-empty,
        # so an oversized lone turn still forms its own chunk.
        breaks = cur and cur_dur + PAD_S + d > max_chunk_s
        # In speaker-coherent mode, a speaker change forces a break too,
        # regardless of remaining room, to keep one speaker per chunk.
        if speaker_coherent and cur and cur[-1].speaker != t.speaker:
            breaks = True
        # Flush the open chunk and reset the accumulator before we place
        # the current turn into a new one.
        if breaks:
            chunks.append(cur)
            cur, cur_dur = [], 0.0
        # Place the turn and grow the running duration. The pad only
        # counts once the chunk already holds at least one turn.
        cur.append(t)
        cur_dur += (PAD_S if cur_dur else 0.0) + d
    # Emit the trailing chunk still being built at end-of-stream.
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def transcribe(model, pcm: np.ndarray) -> str:
    """Transcribe one PCM slice with the warm whisper model.

    Parameters
    ----------
    model : object
        A loaded ``pywhispercpp`` model whose ``transcribe`` returns
        segment objects carrying a ``.text`` attribute.
    pcm : numpy.ndarray
        Mono 16 kHz ``float32`` samples to transcribe.

    Returns
    -------
    str
        The concatenated, stripped segment texts, or ``""`` for slices
        too short to transcribe reliably.
    """
    # Sub-50 ms slices are dropped: whisper tends to hallucinate tokens
    # on near-empty input, which would poison the WER unfairly.
    if pcm.shape[0] < int(0.05 * 16000):  # < 50 ms — whisper hallucinates
        return ""
    segs = model.transcribe(pcm)
    # Whisper returns per-segment objects; join their trimmed texts into
    # one string and trim once more so callers get a clean transcript.
    return " ".join((s.text or "").strip() for s in segs).strip()


def run_per_segment(model, audio, sr, turns) -> tuple[str, float, int]:
    """Transcribe each turn with its own whisper call (today's baseline).

    Mirrors :class:`vocal_helper.asr.WhisperStage`: one call per turn,
    awaited before the next. This is the strategy the batched variants
    are measured against.

    Parameters
    ----------
    model : object
        Warm whisper model (see :func:`transcribe`).
    audio : numpy.ndarray
        Whole-clip mono PCM.
    sr : int
        Sample rate of ``audio`` in Hz.
    turns : list of Turn
        Ground-truth turns to transcribe individually.

    Returns
    -------
    tuple of (str, float, int)
        The joined hypothesis transcript, the whisper wall-time in
        seconds, and the number of whisper calls (== number of turns).
    """
    parts: list[str] = []
    # Time only the transcription loop so the RTF reflects inference, not
    # slicing or setup.
    t0 = time.perf_counter()
    # One whisper call per turn — this is the cost the batched variants
    # try to beat by amortising the fixed 30 s encoder pass.
    for t in turns:
        parts.append(transcribe(model, slice_pcm(audio, sr, t.t0, t.t1)))
    wall = time.perf_counter() - t0
    # Drop empty parts (short/skipped turns) before joining so the hyp is
    # a clean, contiguous transcript. n_calls == len(turns).
    return " ".join(p for p in parts if p).strip(), wall, len(turns)


def run_batched(
    model, audio, sr, turns, max_chunk_s, *, speaker_coherent=False
) -> tuple[str, float, int]:
    """Pack turns into chunks and run one whisper call per chunk.

    The full-throttle alternative to :func:`run_per_segment`: fewer,
    fuller calls amortise whisper's fixed 30 s encode. Turns inside a
    chunk are concatenated with a short silence pad between them.

    Parameters
    ----------
    model : object
        Warm whisper model (see :func:`transcribe`).
    audio : numpy.ndarray
        Whole-clip mono PCM.
    sr : int
        Sample rate of ``audio`` in Hz.
    turns : list of Turn
        Ground-truth turns to pack and transcribe.
    max_chunk_s : float
        Chunk-duration cap handed to :func:`pack_chunks`, in seconds.
    speaker_coherent : bool, optional
        Forwarded to :func:`pack_chunks`; keeps one speaker per chunk
        when ``True`` (default ``False``).

    Returns
    -------
    tuple of (str, float, int)
        The joined hypothesis transcript, the whisper wall-time in
        seconds, and the number of whisper calls (== number of chunks).
    """
    # Group the turns first; each group becomes exactly one whisper call.
    chunks = pack_chunks(turns, max_chunk_s, speaker_coherent=speaker_coherent)
    # A single reusable silence buffer stitched between turns so decoding
    # sees a brief gap at each turn boundary instead of an abrupt splice.
    pad = np.zeros(int(PAD_S * sr), dtype=np.float32)
    parts: list[str] = []
    # Time only the transcription loop (see run_per_segment) for a fair RTF.
    t0 = time.perf_counter()
    for chunk in chunks:
        # Rebuild each chunk's audio: turn, pad, turn, pad, … with the
        # pad inserted before every turn except the first.
        pieces: list[np.ndarray] = []
        for i, t in enumerate(chunk):
            if i:
                pieces.append(pad)
            pieces.append(slice_pcm(audio, sr, t.t0, t.t1))
        # One whisper call for the whole concatenated chunk.
        parts.append(transcribe(model, np.concatenate(pieces)))
    wall = time.perf_counter() - t0
    # Same clean-join as the baseline; n_calls == len(chunks) here.
    return " ".join(p for p in parts if p).strip(), wall, len(chunks)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    """Run the batching sweep over the corpus and print/persist results.

    Parses CLI arguments, loads a single warm whisper model, runs the
    per-segment baseline and every ``batch_*`` / ``batch_spk_*`` variant
    on each clip, then prints pooled-median WER / RTF / call-count tables
    and dumps the same numbers to a JSON sidecar.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If no ``--data-root`` is given, or no ``mix_*.wav`` clips are
        found under it.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    # Data root is off-repo (reference audio is not shipped); accept it
    # via flag or the VH_STUDY_DATA_ROOT env var so CI/headless runs work.
    ap.add_argument(
        "--data-root",
        default=os.environ.get("VH_STUDY_DATA_ROOT", ""),
        help="dir containing bagarre-rich/mix_*.wav + .rttm",
    )
    ap.add_argument("--n", type=int, default=12, help="clips to sample")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    # Comma-separated so several chunk caps can be swept in one run.
    ap.add_argument(
        "--chunk-sizes",
        default=",".join(str(s) for s in DEFAULT_CHUNK_SIZES_S),
        help="comma-separated max_chunk_s values",
    )
    args = ap.parse_args()

    # Fail fast with a clear message if the off-repo corpus wasn't pointed at.
    if not args.data_root:
        raise SystemExit(
            "no --data-root ; pass a dir with bagarre-rich/ (or set VH_STUDY_DATA_ROOT)"
        )
    # Resolve the corpus dir and take at most --n clips (sorted for a
    # stable, reproducible sample across runs).
    rich = Path(args.data_root).expanduser() / "bagarre-rich"
    wavs = sorted(rich.glob("mix_*.wav"))[: args.n]
    if not wavs:
        raise SystemExit(f"no mix_*.wav under {rich}")
    # Parse the sweep points once up front.
    chunk_sizes = [float(x) for x in args.chunk_sizes.split(",")]

    # Truncate any prior transcript, then write the run header.
    _LOG.write_text("")
    log("# Offline ASR batching — per-segment vs concatenated chunks")
    log(f"# model={args.model} threads={args.threads} clips={len(wavs)} chunks={chunk_sizes}")

    # One warm model, reused across every strategy/clip (measures inference,
    # not load) — exactly how WhisperStage holds a single lazy instance.
    from pywhispercpp.model import Model  # type: ignore

    model = Model(args.model, n_threads=args.threads, print_realtime=False, print_progress=False)

    # jiwer computes WER; install it on demand so a fresh study box still
    # runs without a manual pip step.
    try:
        from jiwer import wer
    except ImportError:
        import subprocess

        log("# installing jiwer …")
        subprocess.run(["pip", "install", "-q", "jiwer"], check=True)
        from jiwer import wer

    # Build the strategy roster: the baseline plus, for each chunk cap,
    # both the free-packing and speaker-coherent (_spk) variants.
    strategies = ["per_segment"]
    for s in chunk_sizes:
        strategies.append(f"batch_{s:g}s")
        strategies.append(f"batch_spk_{s:g}s")
    # Per-strategy accumulator of (WER, RTF, n_calls) tuples, one per clip.
    rows: dict[str, list[tuple[float, float, int]]] = {s: [] for s in strategies}

    for wav in wavs:
        # Each clip needs its paired word-level RTFM for ground truth.
        rttm = wav.with_suffix(".rttm")
        if not rttm.exists():
            log(f"{wav.name} : no rttm, skip")
            continue
        # Load audio + derive the shared ground-truth turns and reference
        # transcript once; every strategy on this clip reuses them.
        audio, sr = read_mono(wav)
        dur = audio.shape[0] / sr
        words = parse_rttm_words(rttm)
        turns = turns_from_words(words)
        ref = reference_text(words)
        log(f"\n{wav.name}  dur={dur:.0f}s  turns={len(turns)}  ref_words={len(ref.split())}")

        # Baseline first: WER falls back to 1.0 when either side is empty
        # so a degenerate clip can't spuriously score a perfect 0.
        hyp, wall, calls = run_per_segment(model, audio, sr, turns)
        w = float(wer(ref, hyp)) if ref and hyp else 1.0
        # RTF is whisper wall-time normalised by clip duration.
        rows["per_segment"].append((w, wall / dur, calls))
        log(f"  {'per_segment':<12s}  WER={w:.3f}  RTF={wall / dur:.3f}  calls={calls}")

        # Then every batched variant: both packing modes at each cap.
        for s in chunk_sizes:
            for coherent, tag in ((False, ""), (True, "_spk")):
                hyp, wall, calls = run_batched(
                    model, audio, sr, turns, s, speaker_coherent=coherent
                )
                w = float(wer(ref, hyp)) if ref and hyp else 1.0
                key = f"batch{tag}_{s:g}s"
                rows[key].append((w, wall / dur, calls))
                log(f"  {key:<14s}  WER={w:.3f}  RTF={wall / dur:.3f}  calls={calls}")

    # ----- pooled medians -----
    # Medians (not means) so a single pathological clip can't dominate the
    # verdict; print the summary table header.
    log("\n" + "=" * 60)
    log(f"{'strategy':<14s}  {'med_WER':>8s}  {'med_RTF':>8s}  {'med_calls':>9s}")
    log("-" * 60)
    pooled: dict[str, tuple[float, float, float]] = {}
    for s in strategies:
        vals = rows[s]
        # A strategy with no scored clips (e.g. all RTTMs missing) is skipped.
        if not vals:
            continue
        # Median each metric independently across the clip sample.
        mw = statistics.median(v[0] for v in vals)
        mr = statistics.median(v[1] for v in vals)
        mc = statistics.median(v[2] for v in vals)
        pooled[s] = (mw, mr, mc)
        log(f"{s:<14s}  {mw:>8.3f}  {mr:>8.3f}  {mc:>9.0f}")

    # Head-to-head deltas against the baseline drive the default choice:
    # a variant wins only if it is faster without regressing WER.
    if "per_segment" in pooled:
        base_wer, base_rtf, _ = pooled["per_segment"]
        log("\nvs per_segment baseline :")
        for s, (mw, mr, _) in pooled.items():
            # Skip comparing the baseline with itself.
            if s == "per_segment":
                continue
            # Speedup is baseline RTF over variant RTF; guard a zero RTF.
            speedup = base_rtf / mr if mr else float("inf")
            log(
                f"  {s:<12s}  RTF {speedup:.1f}× faster  "
                f"WER {mw - base_wer:+.3f} ({'better' if mw <= base_wer else 'worse'})"
            )

    # Persist the full result (config + pooled + per-clip) as JSON so the
    # run can be re-plotted / compared without re-running whisper.
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
