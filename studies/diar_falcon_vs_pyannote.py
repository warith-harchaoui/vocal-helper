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

AMI_ROOT = Path("/Users/warithharchaoui/pasdebonneoudemauvaisesituation/data/ami/dev-slice")
MEETINGS = ["IS1008a", "ES2011a"]
SR = 16_000
DEFAULT_LOG = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs/vocal_helper_diar_falcon_2026-06-30.log"
)


def log(msg: str) -> None:
    """Echo a study line to stdout and append it to the on-disk log.

    Parameters
    ----------
    msg : str
        Line to emit. Written verbatim to the console and mirrored to
        :data:`DEFAULT_LOG` so the full run survives after the terminal
        scrolls away.

    Returns
    -------
    None
    """
    # stdout is the live view; the flush keeps ordering sane when the
    # study also writes big JSON blobs to the same terminal.
    print(msg, flush=True)
    # Append (never truncate) so successive log() calls accumulate into
    # one durable transcript of the whole run.
    with open(DEFAULT_LOG, "a") as f:
        f.write(msg + "\n")


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
    Both diarization backends expect a single-channel signal, so any
    multi-channel input is collapsed by averaging across channels.
    """
    # soundfile decodes to float32 directly, avoiding a later int→float cast.
    audio, sr = sf.read(str(path), dtype="float32")
    # AMI mixes are sometimes stored multi-channel; fold to mono by
    # averaging so the backends see one channel regardless of source.
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


def load_reference_annotation(rttm: Path):
    """Build a pyannote reference ``Annotation`` from a word-level RTTM.

    The AMI reference is stored one row per word, which would give a
    reference riddled with sub-second gaps. We bridge consecutive words
    from the same speaker separated by ≤ 200 ms into a single turn so the
    reference matches the turn-level granularity the diarizers emit.

    Parameters
    ----------
    rttm : Path
        Path to a word-level RTTM file (SPEAKER rows, one per word).

    Returns
    -------
    pyannote.core.Annotation
        Reference annotation with one segment per bridged speaker turn.

    Notes
    -----
    Imported lazily so the module still imports on machines without
    ``pyannote`` installed (Falcon-only or metrics-only runs).
    """
    from pyannote.core import Annotation, Segment

    # Parse RTTM into (start, end, speaker) triples. RTTM columns:
    # p[3]=onset, p[4]=duration, p[7]=speaker label.
    rows: list[tuple[float, float, str]] = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        # Skip comments / malformed rows and any non-SPEAKER record type.
        if len(p) < 10 or p[0] != "SPEAKER":
            continue
        t0 = float(p[3])
        rows.append((t0, t0 + float(p[4]), p[7]))
    # Sort chronologically so the bridging pass below can look only at
    # the previous turn of each speaker.
    rows.sort()

    # Bridge adjacent same-speaker words into turns, keyed by speaker.
    by: dict[str, list[list[float]]] = {}
    for t0, t1, spk in rows:
        # Guard against zero / negative-length words in the reference.
        if t1 <= t0:
            continue
        b = by.setdefault(spk, [])
        # Same speaker, gap ≤ 200 ms → extend the open turn instead of
        # starting a new one (this is the "speaker bridging" step).
        if b and t0 - b[-1][1] <= 0.20:
            b[-1][1] = max(b[-1][1], t1)
        else:
            b.append([t0, t1])

    # Materialize the bridged turns into a pyannote Annotation.
    ann = Annotation()
    for spk, turns in by.items():
        for t0, t1 in turns:
            ann[Segment(t0, t1)] = spk
    return ann


def hypothesis_to_annotation(segs: list[tuple[float, float, str]]):
    """Convert a list of diarizer segments into a pyannote ``Annotation``.

    Parameters
    ----------
    segs : list[tuple[float, float, str]]
        Hypothesis segments as ``(start, end, speaker)`` triples, as
        returned by :func:`run_falcon` / :func:`run_pyannote`.

    Returns
    -------
    pyannote.core.Annotation
        Hypothesis annotation ready to feed the DER metric.

    Notes
    -----
    ``pyannote`` is imported lazily for the same reason as in
    :func:`load_reference_annotation`.
    """
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for t0, t1, spk in segs:
        # Drop empty / inverted segments — they would raise inside pyannote.
        if t1 <= t0:
            continue
        # Force the speaker tag to str so mixed int/str tags compare cleanly.
        ann[Segment(t0, t1)] = str(spk)
    return ann


# ---------------------------------------------------------------------------
# Backend wrappers.
# ---------------------------------------------------------------------------


def run_falcon(audio: np.ndarray, sr: int) -> tuple[list[tuple[float, float, str]], float]:
    """Diarize a waveform with Picovoice Falcon and time the call.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono ``float32`` waveform.
    sr : int
        Sample rate of ``audio`` in Hz.

    Returns
    -------
    tuple[list[tuple[float, float, str]], float]
        The hypothesis segments as ``(start, end, speaker)`` triples and
        the wall-clock time in seconds spent inside Falcon.

    Raises
    ------
    RuntimeError
        If ``PICOVOICE_ACCESS_KEY`` is not set in the environment.

    Notes
    -----
    ``pvfalcon`` is imported lazily so the module loads even when Falcon
    is not installed; the caller catches failures and skips Falcon.
    """
    import tempfile
    import pvfalcon  # type: ignore

    # Falcon is gated behind a (free-tier) access key; fail loudly with a
    # signup pointer rather than emitting a cryptic SDK error.
    access_key = os.environ.get("PICOVOICE_ACCESS_KEY", "")
    if not access_key:
        raise RuntimeError(
            "PICOVOICE_ACCESS_KEY is not set in the environment. "
            "Sign up at https://console.picovoice.ai/ (free tier) and "
            "export the key."
        )

    falcon = pvfalcon.create(access_key=access_key)
    # Falcon's Python API reads from a file, so stage the waveform to a
    # temp PCM_16 WAV (auto-deleted on context exit).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, sr, subtype="PCM_16")
        # Time only the inference call, not the file write / model setup.
        t0 = time.perf_counter()
        segs = falcon.process_file(tmp.name)
        wall = time.perf_counter() - t0
    # Release the native handle promptly — Falcon holds a C context open.
    falcon.delete()
    # ``segs`` is a list of namedtuples with ``start_sec``, ``end_sec``, ``speaker_tag``.
    # Prefix the numeric tag with "S" so speaker labels are strings, like pyannote's.
    out = [(s.start_sec, s.end_sec, f"S{s.speaker_tag}") for s in segs]
    return out, wall


def run_pyannote(
    audio: np.ndarray, sr: int, hf_token: str | None
) -> tuple[list[tuple[float, float, str]], float]:
    """Diarize a waveform with pyannote 3.1 and time the call.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono ``float32`` waveform.
    sr : int
        Sample rate of ``audio`` in Hz.
    hf_token : str or None
        Hugging Face access token used to download the gated
        ``pyannote/speaker-diarization-3.1`` pipeline.

    Returns
    -------
    tuple[list[tuple[float, float, str]], float]
        The hypothesis segments as ``(start, end, speaker)`` triples and
        the wall-clock time in seconds spent inside the pipeline.

    Notes
    -----
    ``torch`` and ``pyannote.audio`` are imported lazily so the module
    loads without them; the caller catches failures and skips pyannote.
    """
    import torch
    from pyannote.audio import Pipeline

    # pyannote.audio renamed the HF auth kwarg between major versions
    # — try the new name first and fall back so the bench works
    # against both 3.x and 4.x.
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    # pyannote wants a (channel, samples) tensor; unsqueeze adds the
    # leading channel axis to our mono waveform.
    wave = torch.from_numpy(audio).unsqueeze(0)
    # Time only inference — model loading above is excluded from the RTF.
    t0 = time.perf_counter()
    ann = pipe({"waveform": wave, "sample_rate": sr})
    wall = time.perf_counter() - t0
    # Flatten the Annotation back to (start, end, speaker) triples so both
    # backends return the same shape for the shared DER path.
    out = [(seg.start, seg.end, str(spk)) for seg, _track, spk in ann.itertracks(yield_label=True)]
    return out, wall


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Falcon-vs-pyannote DER / compute study and report results.

    Parses CLI arguments, loads each AMI meeting, diarizes it with both
    backends, scores DER against the bridged reference, and prints a
    pooled-median comparison plus a recommendation. Results are also
    persisted as a JSON sidecar next to :data:`DEFAULT_LOG`.

    Returns
    -------
    None

    Notes
    -----
    Missing files, a missing access key, or an uninstalled backend are
    handled gracefully (logged and skipped) so a partial environment
    still yields a partial — rather than empty — comparison.
    """
    # CLI surface: the HF token (for the gated pyannote pipeline) and the
    # set of meetings to run, both overridable but with sensible defaults.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--meetings", nargs="*", default=MEETINGS)
    args = p.parse_args()

    # Start each run from a clean log file, then record the run header.
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text("")
    log("# Falcon vs pyannote — 2026-06-30")
    log(f"# meetings : {args.meetings}")

    # Fall back to the project's settings.yaml when no token was passed
    # on the CLI or in the environment.
    if not args.hf_token:
        # Read settings.yaml as a last resort.
        try:
            from vocal_helper._settings import resolve_hf_token

            args.hf_token = resolve_hf_token(None)
        except Exception:
            # No settings file / helper — pyannote will simply be skipped.
            pass
    # Log credential presence (never the values) so runs are reproducible.
    log(f"# hf_token  : {'<set>' if args.hf_token else '<missing>'}")
    log(f"# pv_key    : {'<set>' if os.environ.get('PICOVOICE_ACCESS_KEY') else '<missing>'}")

    # The DER metric is the one hard dependency: without it there is
    # nothing to measure, so abort early with a clear message.
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError as exc:
        log(f"missing pyannote.metrics : {exc} — abort")
        return

    # Accumulate per-backend results keyed by meeting for the pooled pass.
    per_meeting: dict[str, dict[str, dict]] = {}

    for m in args.meetings:
        # Each meeting ships a mixed-down WAV and a word-level RTTM.
        mdir = AMI_ROOT / m
        wav = mdir / "mix.wav"
        rttm = mdir / "words.rttm"
        # Skip meetings whose assets aren't present rather than crashing.
        if not wav.exists() or not rttm.exists():
            log(f"\n{m} : missing files, skipping")
            continue
        audio, sr_in = read_mono_wav(wav)
        # The whole protocol assumes 16 kHz; assert so a mismatched file
        # fails loudly instead of silently skewing DER / RTF.
        assert sr_in == SR
        dur = audio.shape[0] / SR
        # Build the bridged reference once and count its distinct speakers.
        ref_ann = load_reference_annotation(rttm)
        n_ref_spk = len(set(ref_ann.labels()))
        log(f"\n=== {m} === dur={dur:.0f}s  n_ref_speakers={n_ref_spk}")

        per_meeting[m] = {}

        # ---- Falcon ----
        try:
            log("  Falcon …")
            segs, wall = run_falcon(audio, SR)
            # Canonical DER protocol: 0.25 s forgiveness collar, overlap
            # counted — matches the 2026-06-30 stitch-threshold sweep.
            metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
            der = float(metric(ref_ann, hypothesis_to_annotation(segs)))
            # Real-time factor: inference wall time per second of audio.
            rtf = wall / dur
            per_meeting[m]["falcon"] = {
                "der": der,
                "wall": wall,
                "rtf": rtf,
                "n_segs": len(segs),
            }
            log(f"    falcon   DER={der:.3f}  RTF={rtf:.3f}  wall={wall:.1f}s  n_segs={len(segs)}")
        except Exception as exc:  # noqa: BLE001
            # One backend failing must not abort the other; record the error.
            log(f"    falcon FAILED : {exc!r}")
            per_meeting[m]["falcon"] = {"error": repr(exc)}

        # ---- pyannote ----
        try:
            log("  pyannote 3.1 …")
            segs, wall = run_pyannote(audio, SR, args.hf_token)
            # Fresh metric per call — DER accumulates state across calls.
            metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
            der = float(metric(ref_ann, hypothesis_to_annotation(segs)))
            rtf = wall / dur
            per_meeting[m]["pyannote"] = {
                "der": der,
                "wall": wall,
                "rtf": rtf,
                "n_segs": len(segs),
            }
            log(f"    pyannote DER={der:.3f}  RTF={rtf:.3f}  wall={wall:.1f}s  n_segs={len(segs)}")
        except Exception as exc:  # noqa: BLE001
            # Same isolation as Falcon above: log and carry on.
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
        # Gather only the meetings where this backend produced a score
        # (failed / skipped meetings carry an "error" key instead).
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
        # Nothing to pool for this backend — skip it entirely.
        if not ders:
            continue
        # Median (not mean) so a single pathological meeting doesn't
        # dominate the pooled comparison across the tiny dev-slice.
        pooled[backend] = {
            "der": statistics.median(ders),
            "rtf": statistics.median(rtfs),
        }
        log(f"{backend:<10s}  {pooled[backend]['der']:>8.3f}  {pooled[backend]['rtf']:>8.3f}")

    # Only compare head-to-head when both backends actually produced numbers.
    if "falcon" in pooled and "pyannote" in pooled:
        # Positive gap = Falcon is worse (higher DER) than pyannote.
        d_gap = pooled["falcon"]["der"] - pooled["pyannote"]["der"]
        # >1 means Falcon runs faster (lower RTF) than pyannote.
        r_speedup = pooled["pyannote"]["rtf"] / pooled["falcon"]["rtf"]
        log(f"\nDER gap (falcon − pyannote) : {d_gap:+.3f}")
        log(f"RTF speedup (falcon vs pyannote) : {r_speedup:.1f}×")
        # Decision rule : if Falcon's DER is within 0.03 of pyannote
        # AND it's ≥ 5× faster, recommend Falcon for the offline
        # path — the compute saving dominates the small quality loss.
        if d_gap <= 0.03 and r_speedup >= 5:
            log(
                "\nRecommendation : adopt Falcon for OfflineDiarStage default — "
                "compute win dominates."
            )
        elif d_gap > 0.03:
            log("\nRecommendation : keep pyannote — Falcon's DER gap is too large here.")
        else:
            log("\nRecommendation : keep pyannote — RTF speedup is marginal.")

    # Persist the full result tree as a JSON sidecar next to the log so
    # downstream tooling can consume the numbers without re-parsing text.
    json_out = DEFAULT_LOG.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "meetings": args.meetings,
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
