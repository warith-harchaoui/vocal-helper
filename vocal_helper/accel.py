"""
vocal_helper.accel
==================

Picking the fastest transcription path this machine can actually run.

Transcription is the one stage whose speed decides whether a **live** session is
possible at all. In batch, a slow engine is an inconvenience; live, the sum of the
stages' real-time factors must stay under 1, or the pipeline takes on a delay it
will never recover and the latency grows without bound.

So this module does not accept a default. It probes the hardware, picks the best
accelerator compiled into the installed whisper.cpp, sizes the thread pool to the
cores that are actually fast, and **says what it chose** — because an unexplained
real-time factor is the hardest kind of performance problem to chase.

The ladder, measured and sourced
--------------------------------
Measured on 60 s of real French speech, Apple Silicon, ``large-v3-turbo-q5``
(2026-11-01, ``studies/asr_acceleration_rtf.md``):

===================================  ======  ==============
path                                  RTF     vs CPU
===================================  ======  ==============
Apple Neural Engine (ANEForge)        —       ~1.0–1.3× over Metal, ~5× less energy
Metal (GPU)                           0.033   **4.9× faster**
CPU + BLAS                            0.164   baseline
===================================  ======  ==============

The ANE row carries no local measurement because the installed build has no ANE
and no CoreML compiled in; its figures are the ones reported upstream, where the
backend was merged on 2026-09-18 (``ggml-org/whisper.cpp#3905``). CoreML sits
between the two at roughly 2× the Metal encoder, and is likewise absent here.

Two findings worth keeping, both counter-intuitive
--------------------------------------------------
**MLX is not the fastest thing on Apple's own hardware.** whisper.cpp with Metal
beat mlx-whisper by 1.85× on the same model and audio. Where the products still
use mlx-whisper, it is for its word-onset accuracy, not its speed.

**"Faster-whisper" is 20× slower here.** CTranslate2 has no Metal path, so it
stays on the CPU: real-time factor 0.657, leaving 1.5× of headroom, which does not
hold a live session. The published "up to 4× faster than whisper.cpp" claims do
not survive contact with Apple Silicon.

Threads: the fast cores, not all of them
----------------------------------------
Apple Silicon has performance and efficiency cores. Handing whisper.cpp a thread
count that spills onto the efficiency cores makes it **slower**, because the fast
cores then wait at every synchronisation point for the slow ones. The default was
a hard-coded 6, which on a 12-core machine with 8 performance cores leaves two of
them idle, and on a 4-performance-core laptop oversubscribes by half.

Usage example
-------------
>>> hw = probe()
>>> hw.performance_threads >= 1
True
>>> plan = resolve("auto")
>>> plan.accelerator in ("ane", "coreml", "metal", "cpu")
True
>>> plan.describe().startswith(plan.accelerator)
True
>>> resolve("cpu").accelerator
'cpu'
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess
from dataclasses import dataclass, field

__all__ = [
    "ACCELERATORS",
    "AccelPlan",
    "Hardware",
    "probe",
    "resolve",
]

ACCELERATORS = ("auto", "ane", "coreml", "metal", "cpu")
"""Accepted values. ``"auto"`` resolves to the best the machine can actually run."""

# Best first. Resolution walks this and takes the first one available.
_LADDER = ("ane", "coreml", "metal")


@dataclass(frozen=True)
class Hardware:
    """What this machine offers, as far as transcription is concerned."""

    system: str
    """``"Darwin"``, ``"Linux"``, ``"Windows"``."""

    machine: str
    """Processor architecture, e.g. ``"arm64"``."""

    apple_silicon: bool
    performance_threads: int
    """Cores worth giving to whisper.cpp — see the module docstring on why not all."""

    whisper_backends: frozenset[str] = field(default_factory=frozenset)
    """Which of ``{"metal", "coreml", "ane", "cuda"}`` the installed build carries."""

    def describe(self) -> str:
        """One line naming the machine and what it can run.

        Returns
        -------
        str
            e.g. ``"Darwin/arm64, 8 fast threads, backends: metal"``.

        Examples
        --------
        >>> "threads" in probe().describe()
        True
        """
        backends = ", ".join(sorted(self.whisper_backends)) or "none"
        return (
            f"{self.system}/{self.machine}, {self.performance_threads} fast threads, "
            f"backends: {backends}"
        )


def _sysctl(name: str) -> int | None:
    """Read one integer ``sysctl``, or ``None`` when it is unavailable."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=2, check=True
        ).stdout.strip()
        return int(out) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _performance_threads(apple_silicon: bool) -> int:
    """How many threads whisper.cpp should get on this machine.

    On Apple Silicon this is the **performance** core count: adding efficiency
    cores makes the stage slower, because every synchronisation point then waits
    on the slowest participant. Elsewhere it is the physical core count, since
    hyper-threaded siblings share the arithmetic units a transcoder saturates.

    Returns
    -------
    int
        At least 1, and at most 16 — past that whisper.cpp's own scaling flattens
        and the extra threads only add contention.
    """
    if apple_silicon:
        fast = _sysctl("hw.perflevel0.physicalcpu")
        if fast:
            return max(1, min(16, fast))
    physical = _sysctl("hw.physicalcpu")
    if physical:
        return max(1, min(16, physical))
    return max(1, min(16, (os.cpu_count() or 2) // 2))


def _whisper_backends() -> frozenset[str]:
    """Which accelerators the installed whisper.cpp build actually carries.

    Probes the shipped shared libraries rather than trusting a version number: a
    wheel built without Metal looks identical from the outside and silently costs
    a factor of five.

    Returns
    -------
    frozenset of str
        Any of ``"metal"``, ``"coreml"``, ``"ane"``, ``"cuda"``.
    """
    found: set[str] = set()
    try:
        import pywhispercpp
    except ImportError:
        return frozenset()

    from pathlib import Path

    root = Path(pywhispercpp.__file__).resolve().parent
    names = [p.name.lower() for p in root.rglob("*") if p.suffix in (".dylib", ".so", ".dll")]
    blob = " ".join(names)
    if "metal" in blob:
        found.add("metal")
    if "coreml" in blob or "whisper.coreml" in blob:
        found.add("coreml")
    if "aneforge" in blob or "ane" in [n.split(".")[0] for n in names]:
        found.add("ane")
    if "cuda" in blob or "cublas" in blob:
        found.add("cuda")
    return frozenset(found)


@functools.lru_cache(maxsize=1)
def probe() -> Hardware:
    """Describe this machine once, and remember it.

    Returns
    -------
    Hardware
        Cached for the process: the answer cannot change while it runs, and the
        probe shells out to ``sysctl``, which is not free in a hot loop.

    Examples
    --------
    >>> probe() is probe()
    True
    >>> probe().system == platform.system()
    True
    """
    system = platform.system()
    machine = platform.machine()
    apple_silicon = system == "Darwin" and machine in ("arm64", "aarch64")
    return Hardware(
        system=system,
        machine=machine,
        apple_silicon=apple_silicon,
        performance_threads=_performance_threads(apple_silicon),
        whisper_backends=_whisper_backends(),
    )


@dataclass(frozen=True)
class AccelPlan:
    """A resolved decision: which accelerator, how many threads, and how to get it."""

    accelerator: str
    threads: int
    model_kwargs: dict = field(default_factory=dict)
    """Extra keyword arguments for the whisper.cpp model constructor."""

    env: dict[str, str] = field(default_factory=dict)
    """Environment variables that must be set for this path to engage."""

    why: str = ""
    """Why this one, in one clause — so a surprising real-time factor is explainable."""

    def describe(self) -> str:
        """One line, for a log that has to explain itself later.

        Examples
        --------
        >>> line = resolve("cpu").describe()
        >>> line.startswith("cpu, ") and line.endswith("asked for explicitly")
        True
        """
        return f"{self.accelerator}, {self.threads} threads — {self.why}"

    def apply_env(self) -> None:
        """Set this plan's environment variables, without clobbering a caller's.

        A variable already set by whoever started the process wins: an operator who
        exported something did it on purpose, and silently overriding that is how a
        benchmark measures the wrong thing.
        """
        for name, value in self.env.items():
            os.environ.setdefault(name, value)


def resolve(accelerator: str = "auto", *, threads: int | None = None) -> AccelPlan:
    """Decide the transcription acceleration path for this machine.

    Parameters
    ----------
    accelerator : str, optional
        One of :data:`ACCELERATORS`. ``"auto"`` (default) takes the best the
        installed build can run. An explicit value is honoured **even when the
        probe says it is unavailable**, because an override exists precisely for
        the case where the probe is wrong — but the reason recorded on the plan
        says so, so a disappointing result is traceable.
    threads : int, optional
        Override the thread count. ``None`` sizes it to the fast cores.

    Returns
    -------
    AccelPlan
        What to do, and why.

    Raises
    ------
    ValueError
        On an unknown accelerator name, at decision time rather than deep inside a
        model constructor.

    Examples
    --------
    >>> resolve("metal", threads=4).threads
    4
    >>> resolve("cpu").model_kwargs
    {'context_params': {'use_gpu': False}}
    >>> resolve("nvidia")
    Traceback (most recent call last):
        ...
    ValueError: unknown accelerator 'nvidia'; choose from ane, auto, coreml, cpu, metal
    """
    if accelerator not in ACCELERATORS:
        raise ValueError(
            f"unknown accelerator {accelerator!r}; choose from {', '.join(sorted(ACCELERATORS))}"
        )

    hardware = probe()
    count = threads if threads is not None else hardware.performance_threads

    if accelerator == "auto":
        for candidate in _LADDER:
            if candidate in hardware.whisper_backends:
                chosen = candidate
                why = f"best available on {hardware.describe()}"
                break
        else:
            chosen = "cpu"
            why = (
                "no accelerated backend in the installed whisper.cpp build — expect "
                "roughly five times the real-time factor of a Metal build"
            )
    else:
        chosen = accelerator
        why = "asked for explicitly"
        if chosen != "cpu" and chosen not in hardware.whisper_backends:
            why = (
                f"asked for explicitly, but {chosen} is not in the installed build "
                f"({hardware.describe()}) — it will fall back on its own"
            )

    model_kwargs: dict = {}
    env: dict[str, str] = {}
    if chosen == "cpu":
        # whisper.cpp defaults to use_gpu=true, so the CPU path must be asked for.
        model_kwargs["context_params"] = {"use_gpu": False}
    if chosen == "ane":
        # The upstream backend is gated by this variable at build and run time;
        # unset, the binary behaves exactly like stock whisper.cpp.
        env["ANEFORGE_ENCODER"] = "1"

    return AccelPlan(accelerator=chosen, threads=count, model_kwargs=model_kwargs, env=env, why=why)


# --------------------------------------------------------------------------- #
#  Placement: which stage runs on which unit, and what that costs together
# --------------------------------------------------------------------------- #

#: Measured real-time factors per stage and compute unit, on Apple Silicon,
#: ``large-v3-turbo-q5`` plus the diarization backends of
#: :mod:`vocal_helper.router` (2026-11-01, ``studies/asr_acceleration_rtf.md``).
#: A missing combination is one nobody has measured, and is absent on purpose
#: rather than guessed.
MEASURED_RTF: dict[tuple[str, str], float] = {
    ("asr", "metal"): 0.033,
    ("asr", "cpu"): 0.164,
    ("diar_online_nemo", "gpu"): 0.030,
    ("diar_online_sherpa", "cpu"): 0.580,
    ("diar_offline_pyannote", "gpu"): 0.067,
    ("diar_offline_nemo", "gpu"): 0.051,
    ("diar_offline_sherpa", "cpu"): 0.580,
    # Silero is tiny and runs on a fraction of one core; measured as noise, kept
    # explicit so the budget does not pretend the stage is free.
    ("vad_silero", "cpu"): 0.010,
}


@dataclass(frozen=True)
class StageLoad:
    """One pipeline stage, the compute unit it occupies, and its real-time factor."""

    name: str
    unit: str
    """``"cpu"``, ``"gpu"`` (Metal or MPS — the same silicon), or ``"ane"``."""

    rtf: float


@dataclass(frozen=True)
class LiveBudget:
    """Whether a set of concurrently-running stages can hold a live stream."""

    per_unit: dict[str, float]
    """Summed real-time factor on each unit: stages sharing one unit serialise."""

    effective_rtf: float
    """The busiest unit. This, not the total, is what a live pipeline must beat."""

    headroom: float
    """``1 / effective_rtf``: how many times faster than real time the chain runs."""

    bottleneck: str

    def holds(self, *, margin: float = 2.0) -> bool:
        """Whether the chain holds a live stream with room to spare.

        Parameters
        ----------
        margin : float, keyword-only, optional
            Required headroom. The default of 2 is not timidity: a machine under
            other load, a longer utterance, a model a size up, or a thermal
            throttle each eat a factor well under two, and crossing 1 is not a
            graceful degradation — it is a delay that grows for the rest of the
            session.

        Returns
        -------
        bool
        """
        return self.headroom >= margin

    def describe(self) -> str:
        """One line: the busiest unit, the headroom, and where the load sits."""
        load = ", ".join(f"{unit} {rtf:.3f}" for unit, rtf in sorted(self.per_unit.items()))
        return (
            f"effective RTF {self.effective_rtf:.3f} ({self.headroom:.1f}x real time), "
            f"bottleneck {self.bottleneck} — {load}"
        )


def live_budget(stages: list[StageLoad]) -> LiveBudget:
    """Combine concurrent stages into the one number a live pipeline must beat.

    **Not a sum.** In a producer/consumer pipeline the stages run concurrently, so
    two stages on **different** compute units overlap and the chain costs the
    slower of the two, not both. Two stages on the **same** unit do not overlap:
    they queue for it, and there the costs do add.

    So the rule is: add within a unit, take the maximum across units. Treating
    stage real-time factors as simply additive, as the obvious reading does,
    overstates the cost of a well-placed chain and — worse — hides the real
    problem, which is never the total but always one saturated unit.

    This is the lever for a live session: the question is not "is each stage fast
    enough" but "are the expensive stages on different silicon".

    Parameters
    ----------
    stages : list of StageLoad
        The stages that run concurrently on the same audio.

    Returns
    -------
    LiveBudget

    Examples
    --------
    Both heavy stages on the GPU: they serialise, and the CPU sits idle.

    >>> budget = live_budget([
    ...     StageLoad("vad", "cpu", 0.010),
    ...     StageLoad("asr", "gpu", 0.033),
    ...     StageLoad("diar", "gpu", 0.030),
    ... ])
    >>> round(budget.effective_rtf, 3), budget.bottleneck
    (0.063, 'gpu')
    >>> budget.holds()
    True

    Moving diarization off the GPU does **not** help when the alternative is an
    expensive CPU backend: the bottleneck simply changes hands, for the worse.

    >>> worse = live_budget([
    ...     StageLoad("vad", "cpu", 0.010),
    ...     StageLoad("asr", "gpu", 0.033),
    ...     StageLoad("diar", "cpu", 0.580),
    ... ])
    >>> round(worse.effective_rtf, 3), worse.bottleneck
    (0.59, 'cpu')

    With no accelerator at all, everything piles onto one unit and the margin
    collapses to the point where a slower machine decouples:

    >>> cramped = live_budget([
    ...     StageLoad("vad", "cpu", 0.010),
    ...     StageLoad("asr", "cpu", 0.164),
    ...     StageLoad("diar", "cpu", 0.580),
    ... ])
    >>> round(cramped.effective_rtf, 3), cramped.holds()
    (0.754, False)
    """
    per_unit: dict[str, float] = {}
    for stage in stages:
        per_unit[stage.unit] = per_unit.get(stage.unit, 0.0) + stage.rtf
    if not per_unit:
        return LiveBudget({}, 0.0, float("inf"), "none")
    bottleneck = max(per_unit, key=lambda unit: per_unit[unit])
    effective = per_unit[bottleneck]
    headroom = (1.0 / effective) if effective > 0 else float("inf")
    return LiveBudget(per_unit, effective, headroom, bottleneck)
