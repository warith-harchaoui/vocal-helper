"""Whisper STT — initial_prompt × language-lock sweep.

Goal
----
Find the operating point of :class:`vocal_helper.asr.WhisperStage` that
maximises **WER quality** at the **lowest RTF**, holding the model
(``large-v3-turbo-q5_0``) constant.

Two levers we tune :

1. ``language`` — ``"auto"`` (the safe default — whisper runs language
   identification on every clip) vs the explicit ISO code (here
   ``"en"`` for AMI). Locking the language skips LID, saves ~ 5-10 %
   RTF and prevents misclassification on noisy / mixed-language input.
2. ``initial_prompt`` — empty (default) vs a vocabulary-biasing prompt
   tuned to the corpus. Whisper conditions its decoding on the prompt
   so domain words spell correctly and rare technical terms don't get
   normalised away.

Sweep matrix
------------

::

    | language | initial_prompt      |
    |----------|---------------------|
    | auto     | ""                  |  baseline
    | auto     | <AMI bias>          |
    | en       | ""                  |
    | en       | <AMI bias>          |

Four configs, two AMI meetings (IS1008a + ES2011a) — eight runs.

Metric
------

- WER (word error rate) of the transcription vs the words.rttm
  reference, computed with :mod:`jiwer`. Lower is better.
- RTF = wall_time / audio_duration. Lower is better.

We pick the Pareto winner (lowest WER at acceptable RTF) and apply
it to vocal-helper's default config.

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

from vocal_helper.asr import WhisperStage
from vocal_helper.types import DiarizedSegment

AMI_ROOT = Path("/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice")
MEETINGS = ["IS1008a", "ES2011a"]
DEFAULT_MODEL = "large-v3-turbo-q5_0"
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/"
    "vocal_helper_whisper_prompt_lang_2026-06-30.log"
)

# AMI is a design / project-management corpus. Biasing whisper with
# the dominant vocabulary categories avoids the most common
# normalisation errors (e.g. "diarization" → "direction").
AMI_BIAS_PROMPT = (
    "AMI meeting transcript: project kickoff, design discussion, "
    "remote control, marketing plan, industrial design, user "
    "interface, requirements, scope, deliverables, timeline."
)

CONFIGS: list[tuple[str, str, str]] = [
    ("auto-no-prompt", "auto", ""),
    ("auto-bias", "auto", AMI_BIAS_PROMPT),
    ("en-no-prompt", "en", ""),
    ("en-bias", "en", AMI_BIAS_PROMPT),
]


def log(msg: str) -> None:
    """Echo a study line to stdout and append it to the on-disk log.

    Parameters
    ----------
    msg : str
        Line to emit. Printed live and mirrored to :data:`DEFAULT_LOG`
        so the full sweep survives after the terminal scrolls away.

    Returns
    -------
    None
    """
    # stdout is the live view; flush keeps ordering deterministic.
    print(msg, flush=True)
    # Append so successive calls accumulate one durable run transcript.
    with open(DEFAULT_LOG, "a") as f:
        f.write(msg + "\n")


def load_reference(rttm: Path) -> str:
    """Build the WER reference text from a word-level RTTM.

    Parameters
    ----------
    rttm : Path
        Path to a word-level RTTM file (SPEAKER rows, one word each).

    Returns
    -------
    str
        Every reference word concatenated in chronological order — the
        ground-truth string that :mod:`jiwer` scores the hypothesis
        against.
    """
    # Collect (onset, word) pairs; RTTM col p[3]=onset, p[-1]=the word.
    words: list[tuple[float, str]] = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        # Skip comments / malformed rows and non-SPEAKER record types.
        if len(p) < 10 or p[0] != "SPEAKER":
            continue
        # Keep only onset + surface word; the rest of the RTTM is unused here.
        words.append((float(p[3]), p[-1]))
    # Sort by onset so the reference reads in spoken order across speakers.
    words.sort()
    # Join into a single space-separated string — jiwer tokenizes on spaces.
    return " ".join(w for _, w in words)


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as a mono float32 waveform.

    Parameters
    ----------
    path : Path
        Path to the WAV file to load.

    Returns
    -------
    tuple[numpy.ndarray, int]
        The mono ``float32`` waveform and its native sample rate.

    Notes
    -----
    Whisper expects a single-channel signal, so multi-channel input is
    collapsed by averaging across channels.
    """
    # Decode straight to float32 to skip a later int→float conversion.
    audio, sr = sf.read(str(path), dtype="float32")
    # Fold any multi-channel AMI mix down to mono by channel-averaging.
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


def transcribe(
    pcm: np.ndarray,
    sr: int,
    language: str,
    initial_prompt: str,
) -> tuple[str, float]:
    """Transcribe a waveform for one (language, prompt) lever combination.

    Parameters
    ----------
    pcm : numpy.ndarray
        Mono ``float32`` waveform of a whole meeting.
    sr : int
        Sample rate of ``pcm`` in Hz (kept for signature symmetry; the
        model is fixed at :data:`DEFAULT_MODEL`'s native rate).
    language : str
        ISO code to lock decoding to, or ``"auto"`` to let whisper run
        language identification on the clip.
    initial_prompt : str
        Vocabulary-biasing prompt, or ``""`` to disable prompt biasing.

    Returns
    -------
    tuple[str, float]
        The transcript text and the wall-clock decode time in seconds.

    Notes
    -----
    ``pywhispercpp`` is imported lazily so the module loads on machines
    without the native whisper.cpp bindings.
    """
    from pywhispercpp.model import Model  # type: ignore

    # Common decode kwargs; language is only passed when locked so that
    # "auto" falls back to whisper's built-in LID.
    kwargs = {"n_threads": 6, "print_realtime": False, "print_progress": False}
    if language != "auto":
        kwargs["language"] = language
    model = Model(DEFAULT_MODEL, **kwargs)

    # Time only the decode call, excluding model construction above.
    t0 = time.perf_counter()
    # Pass the bias prompt only when non-empty so the empty-prompt config
    # exercises whisper's true default decoding path.
    if initial_prompt:
        segs = model.transcribe(pcm, initial_prompt=initial_prompt)
    else:
        segs = model.transcribe(pcm)
    wall = time.perf_counter() - t0
    # Join whisper's sub-segments into one hypothesis string for WER.
    text = " ".join((s.text or "").strip() for s in segs).strip()
    return text, wall


def main() -> None:
    """Run the whisper prompt × language-lock sweep and report the winner.

    Sweeps the four (language, prompt) configurations in :data:`CONFIGS`
    across every AMI meeting, scores WER and RTF per run, prints the
    pooled-median table, selects the Pareto winner (lowest WER, RTF as
    tie-break), and persists a JSON sidecar.

    Returns
    -------
    None

    Notes
    -----
    Missing meeting assets are skipped rather than fatal, so a partial
    corpus still yields a partial sweep.
    """
    # Only knob is the model id; the sweep matrix itself is fixed by CONFIGS.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()
    # NOTE: --model is logged/serialized for provenance; the transcriber
    # itself pins DEFAULT_MODEL so the sweep isolates language × prompt.

    # Reset the log file for a clean run, then write the header block.
    # Ensure the parent run-logs directory exists before writing.
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")

    # Header echoes the run's fixed parameters for reproducibility.
    log(f"# Whisper prompt × language sweep — 2026-06-30")
    log(f"# model    : {args.model}")
    log(f"# meetings : {MEETINGS}")
    log(f"# bias prompt : {AMI_BIAS_PROMPT!r}")

    # Lazy WER — jiwer is the de facto standard.
    # Auto-install on first run so the study is self-contained.
    try:
        from jiwer import wer
    except ImportError:
        log("# installing jiwer …")
        import subprocess

        subprocess.run(["pip", "install", "-q", "jiwer"], check=True)
        from jiwer import wer

    # Per-meeting results keyed by config label, plus per-meeting durations.
    per_meeting: dict[str, dict[str, tuple[float, float, str]]] = {}
    durations: dict[str, float] = {}

    # Outer loop over meetings; inner loop (below) over the four configs.
    for m in MEETINGS:
        # Each meeting ships a mixed WAV and its word-level RTTM reference.
        mdir = AMI_ROOT / m
        wav = mdir / "mix.wav"
        rttm = mdir / "words.rttm"
        # Skip meetings whose assets are absent instead of crashing.
        if not wav.exists() or not rttm.exists():
            log(f"\n{m} : missing files, skipping")
            continue
        audio, sr = read_mono_wav(wav)
        # Duration drives the RTF denominator below.
        dur = audio.shape[0] / sr
        # Build the WER reference once; reused across all four configs.
        ref = load_reference(rttm)
        # Remember the duration so the JSON sidecar can report it.
        durations[m] = dur
        log(f"\n{m}  dur={dur:.0f}s  ref_words={len(ref.split())}")

        per_meeting[m] = {}
        # Run every (language, prompt) config against this meeting.
        for label, lang, prompt in CONFIGS:
            log(f"  running {label} …")
            # Decode under this lever combo; wall is the pure decode time.
            text, wall = transcribe(audio, sr, lang, prompt)
            # WER vs the reference; RTF = decode time per second of audio.
            w = float(wer(ref, text))
            rtf = wall / dur
            # Stash WER, RTF and the raw hypothesis for the summary + JSON.
            per_meeting[m][label] = (w, rtf, text)
            log(
                f"    {label:<15s}  WER={w:.3f}  RTF={rtf:.3f}  wall={wall:.1f}s  hyp_words={len(text.split())}"
            )

    # ----- pooled summary -----
    # Collapse the per-meeting grid to one median row per config so the
    # winner is picked on corpus-level behaviour, not a single meeting.
    log("\n" + "=" * 64)
    log("Pooled median over meetings")
    log("=" * 64)
    log(f"{'config':<16s}  {'med_WER':>8s}  {'med_RTF':>8s}")
    log("-" * 36)
    pooled: dict[str, tuple[float, float]] = {}
    for label, _, _ in CONFIGS:
        # Collect this config's WER / RTF across every meeting that ran.
        wers = [per_meeting[m][label][0] for m in MEETINGS if m in per_meeting]
        rtfs = [per_meeting[m][label][1] for m in MEETINGS if m in per_meeting]
        # Nothing ran for this config — skip it.
        if not wers:
            continue
        # Median over meetings so one hard meeting doesn't skew the pick.
        # Store (WER, RTF) so the winner selection below can weigh both.
        pooled[label] = (statistics.median(wers), statistics.median(rtfs))
        log(f"{label:<16s}  {pooled[label][0]:>8.3f}  {pooled[label][1]:>8.3f}")

    # Pick : lowest median WER ; break ties by RTF.
    # The tuple key sorts by WER first, then RTF — exactly the Pareto rule.
    winner = min(pooled.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    log(f"\nWinner : {winner[0]}  med_WER={winner[1][0]:.3f}  med_RTF={winner[1][1]:.3f}")

    # Persist model, configs, durations and full result tree as JSON so
    # the winner can be applied to vocal-helper's default config later.
    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "model": args.model,
                "meetings": MEETINGS,
                "durations_s": durations,
                "configs": [
                    {"label": label, "language": lang, "initial_prompt": prompt}
                    for label, lang, prompt in CONFIGS
                ],
                "pooled": {k: list(v) for k, v in pooled.items()},
                "per_meeting": {
                    m: {label: [v[0], v[1]] for label, v in per_meeting[m].items()}
                    for m in per_meeting
                },
                "winner": winner[0],
            },
            indent=2,
        )
    )
    log(f"\nJSON dump : {json_out}")
    log("\n[done]")


if __name__ == "__main__":
    main()
