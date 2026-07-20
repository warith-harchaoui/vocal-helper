"""
Offline diarization + transcription quality regression (DeepEval).

Module summary
--------------
Runs the **HF-free** offline stack — ``OfflineDiarStage(backend="pyannote")``
(loaded from the self-hosted diarization-engines bundle) plus whisper.cpp
via ``transcribe_pcm`` — on a small self-hosted AMI subset, and asserts
that diarization error (DER) and word error (WER) stay within versioned
thresholds. Metrics are expressed as custom DeepEval ``BaseMetric``s so
the AI-evaluation layer (per the project standard) gates the pipeline the
same way unit tests gate plain code.

The test is marked ``integration``: it is skipped by default (it needs the
pyannote weights + whisper model + the hosted subset) and runs only when
``pytest -m integration`` is invoked with the assets available. No
HuggingFace access is required — everything resolves from the bundle.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np
import pytest

# The evaluation layer is optional at import time so the collection step
# never explodes on a machine without deepeval; the test skips instead.
deepeval = pytest.importorskip("deepeval")
from deepeval.metrics import BaseMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

# The AMI fixture is a sibling test-support module. Import it whether the suite
# runs with ``tests/`` on the path as a package (``tests._ami_fixture``) or with
# the tests dir itself on ``sys.path`` (plain ``_ami_fixture``) — and skip the
# whole (integration-only) module if it cannot be found, so mere collection on a
# machine that happens to have deepeval installed never explodes.
try:
    from tests._ami_fixture import load_ami_clips  # noqa: E402
except ModuleNotFoundError:
    try:
        from _ami_fixture import load_ami_clips  # type: ignore[no-redef]  # noqa: E402
    except ModuleNotFoundError:  # pragma: no cover - support module absent
        pytest.skip(
            "tests/_ami_fixture support module not importable (integration-only).",
            allow_module_level=True,
        )

# ---------------------------------------------------------------------------
# Versioned thresholds. Generous margins over the observed pyannote + whisper
# numbers on 60 s AMI clips — this is a *regression* guard (catch a large
# degradation), not a tight benchmark. Bump only with a measured justification.
# ---------------------------------------------------------------------------
MAX_MEDIAN_DER = 0.35  # median diarization error rate across clips
MAX_MEDIAN_WER = 0.55  # median whisper-normalised word error rate
N_CLIPS = 3  # cap clips so the integration run stays quick


class DERMetric(BaseMetric):
    """DeepEval metric wrapping diarization error rate.

    Parameters
    ----------
    threshold : float
        Maximum acceptable DER; the metric succeeds when ``score`` is at
        or below it.

    Notes
    -----
    The reference and hypothesis speaker turns are passed through the test
    case's ``additional_metadata`` (``ref_turns`` / ``hyp_turns``, each a
    list of ``(t0, t1, speaker)``) because DER is not derivable from the
    plain text fields DeepEval models by default.
    """

    def __init__(self, threshold: float) -> None:
        """Store the DER threshold and null-init the DeepEval result fields."""
        # DeepEval reads these attributes after ``measure``.
        self.threshold: float = threshold
        self.score: float | None = None
        self.success: bool | None = None
        self.reason: str | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        """Compute DER from the case's reference / hypothesis turns."""
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        meta: dict[str, Any] = test_case.additional_metadata or {}
        # Rebuild pyannote Annotations from the (t0, t1, speaker) triples.
        ref = Annotation()
        for t0, t1, spk in meta["ref_turns"]:
            ref[Segment(t0, t1)] = spk
        hyp = Annotation()
        for t0, t1, spk in meta["hyp_turns"]:
            hyp[Segment(t0, t1)] = spk
        # 0.25 s forgiveness collar matches the pdbms / literature setup.
        self.score = float(DiarizationErrorRate(collar=0.25)(ref, hyp))
        self.success = self.score <= self.threshold
        self.reason = f"DER {self.score:.3f} (threshold {self.threshold})"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async shim — DER is cheap and synchronous."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Return whether the last ``measure`` met the threshold."""
        return bool(self.success)

    @property
    def __name__(self) -> str:  # noqa: D401 - DeepEval display name
        """Human-readable metric name shown in DeepEval output."""
        return "DER"


class WERMetric(BaseMetric):
    """DeepEval metric wrapping whisper-normalised word error rate.

    Parameters
    ----------
    threshold : float
        Maximum acceptable WER; the metric succeeds at or below it.

    Notes
    -----
    Reads ``actual_output`` (hypothesis transcript) and ``expected_output``
    (reference transcript) from the test case and normalises both with the
    Whisper English normaliser before scoring with ``jiwer``.
    """

    def __init__(self, threshold: float) -> None:
        """Store the WER threshold and null-init the DeepEval result fields."""
        self.threshold: float = threshold
        self.score: float | None = None
        self.success: bool | None = None
        self.reason: str | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        """Compute normalised WER between hypothesis and reference text."""
        import jiwer
        from whisper_normalizer.english import EnglishTextNormalizer

        # Normalise casing / punctuation the same way on both sides.
        norm = EnglishTextNormalizer()
        ref = norm(test_case.expected_output or "")
        hyp = norm(test_case.actual_output or "")
        # An empty reference would make WER undefined — treat as a hard fail.
        self.score = float(jiwer.wer(ref, hyp)) if ref else 1.0
        self.success = self.score <= self.threshold
        self.reason = f"WER {self.score:.3f} (threshold {self.threshold})"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async shim — WER is cheap and synchronous."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Return whether the last ``measure`` met the threshold."""
        return bool(self.success)

    @property
    def __name__(self) -> str:  # noqa: D401 - DeepEval display name
        """Human-readable metric name shown in DeepEval output."""
        return "WER"


def _rttm_turns(path) -> list[tuple[float, float, str]]:
    """Parse an RTTM file into ``(t0, t1, speaker)`` turns.

    Parameters
    ----------
    path : Path
        RTTM file with clip-relative timings.

    Returns
    -------
    list of tuple
        One ``(start_s, end_s, speaker)`` per ``SPEAKER`` line.
    """
    turns: list[tuple[float, float, str]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        # RTTM speaker lines: SPEAKER <uri> 1 <t0> <dur> <NA> <NA> <spk> ...
        if parts and parts[0] == "SPEAKER":
            t0, dur, spk = float(parts[3]), float(parts[4]), parts[7]
            turns.append((t0, t0 + dur, spk))
    return turns


@pytest.mark.integration
def test_offline_der_wer_regression() -> None:
    """Offline stack keeps median DER / WER within versioned thresholds.

    Skips when the hosted AMI subset or the diarization models are not
    available (e.g. offline CI), so the fast suite is never blocked.
    """
    # Fetch a capped, verified slice of the self-hosted subset.
    clips = load_ami_clips(limit=N_CLIPS)
    if not clips:
        pytest.skip("AMI subset unavailable (offline / host unreachable)")

    # Import the heavy stack lazily and skip if the extras aren't installed.
    try:
        import soundfile as sf

        from vocal_helper.asr import transcribe_pcm
        from vocal_helper.diar import OfflineDiarStage
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"offline stack unavailable: {exc!r}")

    # HF-free: OfflineDiarStage resolves pyannote from the bundle. Loading
    # the model is itself the smoke test for the whole HF-removal path.
    try:
        diar = OfflineDiarStage(backend="pyannote")
        diar._ensure_backend()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"pyannote weights unavailable: {exc!r}")

    ders: list[float] = []
    wers: list[float] = []
    # Evaluate each clip end-to-end and collect its two scores.
    for clip in clips:
        # Load the 60 s mono clip and run diarization on the whole buffer.
        pcm, sr = sf.read(str(clip["wav"]), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        pcm = pcm.astype(np.float32)
        hyp_turns = diar.diarize(pcm, sr)

        # Transcribe the same buffer (whole-buffer, English) for WER.
        hyp_text = transcribe_pcm(pcm, sr, language="en")
        ref_text = clip["reference_txt"].read_text()

        # Wrap the clip as a DeepEval case; annotations ride in metadata.
        case = LLMTestCase(
            input=clip["clip_id"],
            actual_output=hyp_text,
            expected_output=ref_text,
            additional_metadata={
                "ref_turns": _rttm_turns(clip["reference_rttm"]),
                "hyp_turns": hyp_turns,
            },
        )
        # Score with the two custom metrics.
        der = DERMetric(MAX_MEDIAN_DER)
        wer = WERMetric(MAX_MEDIAN_WER)
        ders.append(der.measure(case))
        wers.append(wer.measure(case))

    # Aggregate: the median is robust to a single hard clip.
    median_der = statistics.median(ders)
    median_wer = statistics.median(wers)
    assert median_der <= MAX_MEDIAN_DER, f"median DER {median_der:.3f} > {MAX_MEDIAN_DER}"
    assert median_wer <= MAX_MEDIAN_WER, f"median WER {median_wer:.3f} > {MAX_MEDIAN_WER}"
