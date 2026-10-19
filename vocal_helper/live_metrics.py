"""
vocal_helper.live_metrics
=========================

What a live session actually costs the person watching it: how long after the words
were spoken the text appeared, how often what appeared was later rewritten, and where in
the producer/consumer chain the wait accumulated.

Why this module exists
----------------------
The suite already has a precise latency model — ``operator-assistance-streaming``'s
``CADENCE.md`` gives per-stage constants and a table of expected on-screen delays, down to
"a 3-second line appears at t ≈ 3.85 s". It is an excellent model and it is **predicted,
never observed**. Nothing in the live path records what really happened, so a regression in
a queue, a model swap that doubles a round-trip, or a scheduler change that starves a
consumer, all land silently on the one axis a live product is judged on.

Offline quality has WER and DER. Live has these three, and they are not the same question:

``latency``
    Seconds between the end of a piece of speech and the moment its text reached the
    consumer. Measured from the **end** of the utterance, not the start, because that is
    when a listener starts waiting. A long line is not late because it was long.
``revision`` ("scintillement")
    The share of emitted items later rewritten. Text that jumps is unusable at any word
    error rate, and a pipeline that emits early and corrects often can post a flattering
    latency while being worse to watch than one that waits.
``queue wait``
    Where the latency came from. ``CADENCE.md``'s own framing: user-visible latency is the
    sum of waits in the queues plus each stage's compute. Without the split, a latency
    number says something is slow and never which thing.

The invariant
-------------
``CADENCE.md`` states one rule the whole design rests on: *no UI update appears before the
audio that produced it*. :meth:`LiveMetrics.violations` turns that from prose into
something a test can fail on — an event whose audio had not been captured yet when it was
emitted means a clock, a buffer or a mock is lying, and every number computed next to it is
meaningless.

Nothing here touches audio, models or the network: it records timestamps a caller provides,
so it is exercised in tests without a microphone and costs a dictionary insert in
production.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field


def _pct(values: list[float], q: float) -> float | None:
    """``q``-quantile by nearest rank, or ``None`` on no data.

    Nearest rank rather than interpolation: these samples are small (a live session yields
    tens of utterances, not thousands), and an interpolated p95 over twenty points invents a
    number that was never observed. Reporting a real sample is more honest at this size.
    """
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


@dataclass
class _Emission:
    """One thing the pipeline handed the consumer, and when."""

    key: tuple[str, float]
    text: str
    audio_end_s: float
    emitted_at_s: float


@dataclass
class LiveMetrics:
    """Collects live-session timings. One instance per session.

    Parameters
    ----------
    started_at : float, optional
        Monotonic clock reading at the moment audio capture began. Defaults to now, which
        is right when the collector is created alongside the source and wrong if it is
        created later — pass it explicitly when replaying a file, where "now" and "the
        start of the audio" are different instants.

    Examples
    --------
    >>> m = LiveMetrics(started_at=0.0)
    >>> m.record(speaker="S0", t0=0.0, t1=3.0, text="bonjour", now_s=3.85)
    >>> round(m.report()["latency_p50_s"], 2)
    0.85
    """

    started_at: float = field(default_factory=time.monotonic)
    _emissions: list[_Emission] = field(default_factory=list)
    _latest: dict[tuple[str, float], _Emission] = field(default_factory=dict)
    _revisions: int = 0
    _queue_waits: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _violations: list[dict] = field(default_factory=list)

    # -- recording ------------------------------------------------------------------- #

    def record(self, *, speaker: str, t0: float, t1: float, text: str,
               now_s: float | None = None) -> None:
        """Record one item reaching the consumer.

        ``t0``/``t1`` are positions in the AUDIO timeline; ``now_s`` is the monotonic clock
        when the item was handed over (defaults to now). Re-recording the same
        ``(speaker, t0)`` with different text counts as a revision — which is what
        flicker is — while identical text is a duplicate delivery and is not.
        """
        now = time.monotonic() if now_s is None else now_s
        elapsed = now - self.started_at
        key = (str(speaker), round(float(t0), 2))
        em = _Emission(key=key, text=str(text), audio_end_s=float(t1), emitted_at_s=now)

        if float(t1) > elapsed + 1e-6:
            # The audio this claims to transcribe had not been captured yet. A clock, a
            # buffer or a mock is lying, and every latency computed beside it is fiction.
            self._violations.append({
                "speaker": speaker, "t0": t0, "t1": t1,
                "elapsed_s": round(elapsed, 3),
                "ahead_by_s": round(float(t1) - elapsed, 3),
            })

        previous = self._latest.get(key)
        if previous is not None and previous.text != em.text:
            self._revisions += 1
        self._latest[key] = em
        self._emissions.append(em)

    def record_queue_wait(self, stage: str, seconds: float) -> None:
        """Record how long one item waited in *stage*'s inbox before being picked up.

        This is the half of latency a total cannot show. ``CADENCE.md``: user-visible
        latency is the sum of queue waits plus per-stage compute, so a slow session with
        no queue wait is a slow model, and the same number with a large queue wait is a
        starved consumer. Opposite fixes.
        """
        if seconds >= 0:
            self._queue_waits[str(stage)].append(float(seconds))

    # -- reading ---------------------------------------------------------------------- #

    @property
    def violations(self) -> list[dict]:
        """Items emitted before their audio existed. Non-empty means do not trust the rest."""
        return list(self._violations)

    def latencies(self) -> list[float]:
        """Per-item seconds between the end of the speech and its delivery.

        From the END of the utterance: that is when a listener starts waiting. Measuring
        from the start would charge a long sentence for being long, and would reward a
        pipeline that chops speech into fragments.
        """
        return [em.emitted_at_s - (self.started_at + em.audio_end_s) for em in self._emissions]

    def revision_rate(self) -> float:
        """Share of deliveries that rewrote something already shown, in ``[0, 1]``.

        Zero is not automatically good: a pipeline that only emits finalised text cannot
        flicker and pays for it in latency instead. The pair is the measurement, never
        either alone.
        """
        return self._revisions / len(self._emissions) if self._emissions else 0.0

    def report(self) -> dict:
        """One flat dict: latency quantiles, revisions, per-queue waits, violations."""
        lat = self.latencies()
        waits = {
            stage: {
                "n": len(xs),
                "p50_s": _pct(xs, 0.5),
                "p95_s": _pct(xs, 0.95),
                "total_s": round(sum(xs), 3),
            }
            for stage, xs in sorted(self._queue_waits.items())
        }
        return {
            "items": len(self._emissions),
            "latency_p50_s": _pct(lat, 0.5),
            "latency_p95_s": _pct(lat, 0.95),
            "latency_max_s": max(lat) if lat else None,
            "latency_mean_s": statistics.fmean(lat) if lat else None,
            "revisions": self._revisions,
            "revision_rate": round(self.revision_rate(), 4),
            "queue_waits": waits,
            "violations": len(self._violations),
        }

    def summary(self) -> str:
        """A one-line human reading of :meth:`report`, for a log or a CLI tail."""
        r = self.report()
        if not r["items"]:
            return "live: aucun élément enregistré"
        parts = [
            f"live: {r['items']} éléments",
            f"latence p50 {r['latency_p50_s']:.2f} s / p95 {r['latency_p95_s']:.2f} s",
            f"révisions {r['revision_rate'] * 100:.0f} %",
        ]
        if r["queue_waits"]:
            worst = max(r["queue_waits"].items(), key=lambda kv: kv[1]["p95_s"] or 0.0)
            parts.append(f"file la plus lente « {worst[0]} » p95 {worst[1]['p95_s']:.2f} s")
        if r["violations"]:
            parts.append(f"⚠️ {r['violations']} émission(s) AVANT leur audio — chiffres non fiables")
        return " · ".join(parts)
