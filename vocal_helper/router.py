"""
vocal_helper.router
===================

The **aiguilleur** — the study-grounded diarization backend router.

Diarization is the one pipeline stage with a genuine backend fork, and there is
**no single best backend**: the right one depends on whether the audio is a
**live stream** or a **batch file**, on its **duration**, on the **speaker
count**, and on whether the deployment can afford a PyTorch install. This module
turns those scenario conditions into one explicit, testable decision that
carries **both quality (DER) and speed (RTF)** so the CLIs and downstream code
never hard-code a backend, re-derive the trade-off, or hide the cost.

Quality × speed per scenario (the whole point)
----------------------------------------------
Numbers were **re-validated on this machine** (2026-07-19,
``studies/router_profile_validation.py``, ``pyannote.metrics`` collar 0.25, median
DER + RTF) against ground truth — bagarre (30 short mixes) + AMI dev-slice (2 real
meetings). ``sherpa`` is from ADR 0002 (its ONNX models are not in the local
bundle). DER = quality (lower is better); RTF = speed (``< 1`` = faster than real
time):

======== ========= ============ ========== =================================================
mode     backend   DER          RTF        when the router picks it
======== ========= ============ ========== =================================================
offline  nemo      **0.142**    0.051      short (≤300 s), ≤4 speakers — dense interleaved turns
offline  pyannote  **0.122**    0.067      long / unknown length / >4 speakers — the robust default
offline  sherpa    0.174        0.58       torch-free deployment (no PyTorch) — ADR 0002
online   nemo      0.586        0.030      any live stream (the default online embedder)
online   sherpa    0.174        0.58       torch-free streaming = periodic offline re-diarization
======== ========= ============ ========== =================================================

Why a router, not a default
---------------------------
Two independent findings, both measured on this machine:

1. **Offline: a length crossover.** On short dense turns (bagarre, ~30 s) NeMo
   Sortformer wins by ~2.3x — offline DER **0.142** vs pyannote **0.330** — its
   end-to-end slot attribution drives speaker-confusion to ~0. On long meetings
   (AMI) the verdict *reverses*: pyannote median DER **0.122** (inside Bredin
   2023's 0.188 band), and Sortformer *hangs* past ~25 min (no output on a 27-min
   meeting) — its 90 s / 4-speaker cap puts long/crowded form out of distribution.
   So offline needs a router: "ship nemo" or "ship pyannote" is wrong for one
   common workload.

2. **Online: nemo, always.** vocal-helper's ``OnlineDiarStage`` is a
   latency-bound cosine-clustering *approximation* — inherently ~3-4x the offline
   DER (it cannot model overlap). Across lengths the NeMo TitaNet embedder is the
   best online backend (bagarre 0.586, AMI 0.497) and beats pyannote/embedding
   online (0.590 / **0.844**), matching the 2026-06-30 embedding sweep that made
   it the default. There is **no online length crossover** — streaming routes to
   nemo (``refine_on_close`` roughly halves the DER on meetings that over-segment,
   a stage knob the router leaves to the stage).

The torch-free ``sherpa`` (ONNX TitaNet-large, DER 0.174/0.148, *beats* NeMo
Sortformer 0.267, FR+EN validated — ADR 0002) is the portability pick either way.

Scope
-----
This module decides the **diarization** backend (mode + backend), the one stage
with a real fork. The other stages are single-backend by study verdict and are
not routed: VAD is Silero v5 (32 ms cadence), ASR is pywhispercpp
``large-v3-turbo`` (no faster-whisper win on CPU/Apple in the study), the
analyst is Gemma via Ollama. Language is *discovered*, never routed to a default
(see :mod:`vocal_helper.lid`).

Usage example
-------------
>>> from vocal_helper.router import select_diarization
>>> plan = select_diarization(live=False, duration_s=45.0, max_speakers=3)
>>> plan.backend, plan.expected_der, plan.expected_rtf
('nemo', 0.142, 0.051)
>>> select_diarization(live=False, duration_s=1800.0).backend
'pyannote'

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

from dataclasses import dataclass

# Offline NeMo Sortformer stays reliable up to ~300 s (it chunks internally at
# its 60 s ideal duration) but *hangs* on very long meetings — the pdbms study
# saw no output on a 27-min AMI file. Past this ceiling, only pyannote is safe.
NEMO_MAX_DURATION_S: float = 300.0

# Sortformer is trained/capped at 4 speakers. Beyond that it silently mislabels,
# so a short clip with a known >4 speaker count must still go to pyannote.
SORTFORMER_MAX_SPEAKERS: int = 4

# Representative (DER, RTF) per (mode, backend), re-validated on this machine
# (2026-07-19; studies/router_profile_validation.py; median DER + RTF). Keyed so
# a decision only names (mode, backend) and the quality + speed numbers follow —
# they can never drift from the reason. DER = quality (lower better); RTF = speed
# (< 1 faster than real time). Only the (mode, backend) pairs the router can emit
# are listed — online never routes to pyannote (it loses to nemo online at every
# length: 0.590/0.844 vs 0.586/0.497).
_PROFILE: dict[tuple[str, str], tuple[float, float]] = {
    ("offline", "nemo"): (0.142, 0.051),  # bagarre n=30, offline Sortformer
    ("offline", "pyannote"): (0.122, 0.067),  # AMI dev-slice n=2 median
    ("offline", "sherpa"): (0.174, 0.58),  # ADR 0002, ES2011a, TitaNet-large ONNX
    ("online", "nemo"): (0.586, 0.030),  # bagarre n=30; AMI long 0.497 — latency-bound ~4x offline
    ("online", "sherpa"): (0.174, 0.58),  # periodic offline re-diarization (ADR 0002)
}


@dataclass(frozen=True)
class BackendPlan:
    """One routing decision: which diarizer, and its quality + speed.

    Attributes
    ----------
    mode : str
        ``"online"`` (streaming :class:`~vocal_helper.OnlineDiarStage`) or
        ``"offline"`` (whole-buffer :class:`~vocal_helper.OfflineDiarStage`).
    backend : str
        Diarization backend — ``"pyannote"``, ``"nemo"`` or ``"sherpa"`` — to
        pass straight to the stage's ``diar={"backend": ...}`` config.
    expected_der : float
        Representative diarization error rate (quality — lower is better) for
        this scenario, from the pdbms study / ADR 0002.
    expected_rtf : float
        Representative real-time factor (speed — ``< 1`` is faster than real
        time) for this scenario.
    reason : str
        Human-readable justification citing the deciding measurement, surfaced
        to the operator so the choice is never a black box.
    """

    mode: str
    backend: str
    expected_der: float
    expected_rtf: float
    reason: str


def _plan(mode: str, backend: str, reason: str) -> BackendPlan:
    """Assemble a :class:`BackendPlan`, attaching quality + speed from ``_PROFILE``.

    Parameters
    ----------
    mode : str
        ``"online"`` or ``"offline"``.
    backend : str
        ``"pyannote"``, ``"nemo"`` or ``"sherpa"``.
    reason : str
        The justification string for this decision.

    Returns
    -------
    BackendPlan
        The plan with ``expected_der`` / ``expected_rtf`` looked up for
        ``(mode, backend)`` so the numbers can never contradict the choice.
    """
    # Single source of truth for the scenario's quality/speed — no hand-typed
    # numbers at the call sites that could drift from the reason.
    der, rtf = _PROFILE[(mode, backend)]
    return BackendPlan(
        mode=mode, backend=backend, expected_der=der, expected_rtf=rtf, reason=reason
    )


def select_diarization(
    *,
    live: bool,
    duration_s: float | None = None,
    max_speakers: int | None = None,
    torch_free: bool = False,
    pyannote_available: bool = True,
) -> BackendPlan:
    """Route to the diarization backend that the experiments justify.

    Encodes the pdbms unified-study crossover (see the module docstring) as a
    single, testable decision over the conditions that actually move DER
    (quality) and RTF (speed): live-vs-batch, duration, speaker count, and
    torch-availability. The returned plan carries both axes explicitly.

    Parameters
    ----------
    live : bool
        ``True`` for a live stream (streaming diarizer), ``False`` for a batch
        file (whole-buffer offline diarizer — the reliable default for files).
    duration_s : float or None, optional
        Audio duration in seconds when known (a file's length is cheap to
        read). ``None`` means "unknown" and is treated as long-form — the safe,
        robust branch. Default ``None``.
    max_speakers : int or None, optional
        Known upper bound on the number of speakers. Used only to keep audio
        with more than :data:`SORTFORMER_MAX_SPEAKERS` speakers off NeMo
        Sortformer (its hard 4-speaker cap). ``None`` = unknown. Default
        ``None``.
    torch_free : bool, optional
        ``True`` when the deployment cannot install PyTorch — routes to the
        ``sherpa`` onnxruntime backend regardless of length. Default ``False``.
    pyannote_available : bool, optional
        Whether the pyannote backend can actually run (extra installed + bundle
        present). When a rule would pick ``pyannote`` but it is unavailable, the
        router falls back rather than choosing an unrunnable backend. Default
        ``True``.

    Returns
    -------
    BackendPlan
        The chosen ``mode`` + ``backend``, its ``expected_der`` /
        ``expected_rtf``, and the ``reason``.

    Examples
    --------
    >>> select_diarization(live=False, duration_s=45.0, max_speakers=3).backend
    'nemo'
    >>> select_diarization(live=False, duration_s=1800.0).backend
    'pyannote'
    >>> select_diarization(live=True, torch_free=True).backend
    'sherpa'

    Notes
    -----
    The router decides *which* diarizer and reports its scenario quality/speed;
    the online/offline *stages* keep their own tuned knobs (join threshold,
    refine pass, chunk ceiling). Numbers were re-validated on this machine
    (``studies/router_profile_validation.py``, 2026-07-19) against ground truth;
    sherpa is from ADR 0002.
    """
    mode = "online" if live else "offline"

    # 1. Portability override — no PyTorch available. The torch-free ONNX
    #    TitaNet (sherpa) is the only runnable backend; quality beats NeMo
    #    Sortformer, so it is a safe forced pick. For streaming, sherpa runs as
    #    periodic offline re-diarization (ADR 0002: per-segment online sherpa is
    #    a dead end), which is why online sherpa carries the offline DER 0.174.
    if torch_free:
        return _plan(
            mode,
            "sherpa",
            "torch-free deployment → sherpa (onnxruntime TitaNet-large, no PyTorch); "
            "DER 0.174 ES2011a / 0.148 held-out IS1008a, beats NeMo Sortformer 0.267, "
            "FR+EN validated (ADR 0002)"
            + ("; streaming = periodic offline re-diarization" if live else ""),
        )

    # 2. STREAMING → nemo, always. vocal-helper's OnlineDiarStage is a
    #    latency-bound cosine-clustering approximation (~3-4x the offline DER);
    #    the NeMo TitaNet embedder is the best online backend at *every* length
    #    measured here (bagarre 0.586, AMI 0.497) and beats pyannote/embedding
    #    online (0.590 / 0.844) — the 2026-06-30 default. There is no online
    #    length crossover, and the 4-speaker cap is Sortformer/offline-only (the
    #    online path uses TitaNet embeddings, uncapped). refine_on_close (a stage
    #    knob) roughly halves the DER on meetings that over-segment.
    if live:
        der, rtf = _PROFILE[("online", "nemo")]
        return _plan(
            "online",
            "nemo",
            f"live stream → nemo TitaNet embedder (best online backend at every "
            f"length; DER {der}, RTF {rtf} — online is a latency-bound ~3-4x-offline "
            f"approximation, refine_on_close helps long meetings)",
        )

    # 3. OFFLINE short / dense regime — NeMo Sortformer wins by ~2.3x on short
    #    interleaved turns (DER 0.142 vs pyannote 0.330 on bagarre). Its 4-speaker
    #    cap means a *known* larger count must skip it, and "unknown duration" is
    #    deliberately NOT short — without a length we take the robust branch.
    too_many_speakers = max_speakers is not None and max_speakers > SORTFORMER_MAX_SPEAKERS
    short_enough = duration_s is not None and duration_s <= NEMO_MAX_DURATION_S
    if short_enough and not too_many_speakers:
        der, rtf = _PROFILE[("offline", "nemo")]
        return _plan(
            "offline",
            "nemo",
            f"≤{NEMO_MAX_DURATION_S:.0f}s, ≤{SORTFORMER_MAX_SPEAKERS} speakers → "
            f"nemo Sortformer (end-to-end, confusion ~0; DER {der}, RTF {rtf})",
        )

    # 4. OFFLINE long-form / unknown / many-speaker → pyannote, the robust
    #    default (AMI median DER 0.122); NeMo hangs past ~25 min and caps at 4
    #    speakers, so pyannote is the only safe choice here.
    if pyannote_available:
        why_long = (
            "unknown duration (treated as long-form)"
            if duration_s is None
            else f">{NEMO_MAX_DURATION_S:.0f}s"
            if not short_enough
            else f">{SORTFORMER_MAX_SPEAKERS} speakers"
        )
        der, rtf = _PROFILE[("offline", "pyannote")]
        return _plan(
            "offline",
            "pyannote",
            f"{why_long} → pyannote 3.1 (robust default; DER {der}, RTF {rtf}; "
            f"nemo hangs past ~25 min / >4 speakers)",
        )

    # 5. pyannote was the right call but is not installed/bundled — fall back to
    #    the torch-free sherpa rather than an unrunnable backend. (nemo is unsafe
    #    here: this branch is exactly the long/many-speaker case it fails on.)
    return _plan(
        "offline",
        "sherpa",
        "pyannote unavailable on the long-form/robust branch → sherpa "
        "(onnxruntime TitaNet); nemo is unsafe past ~25 min / >4 speakers",
    )
