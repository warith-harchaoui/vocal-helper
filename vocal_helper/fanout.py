"""
vocal_helper.fanout
===================

Fork one queue into several, so a producer never has to know who is listening.

Why the pipeline needed this
----------------------------
``Pipeline`` wires its raw-audio queue to exactly one consumer, the voice-activity stage.
That was enough while every diarizer in the package worked on *voiced segments*, and it stops
being enough the moment one works on the audio itself: :class:`~vocal_helper.diar.DiartDiarStage`
consumes PCM directly, and an end-of-stream batch diarizer wants the whole stream. Neither can
be attached without a second reader, and making the source aware of them would put the
knowledge of who consumes what back into the thing that should know least.

Ported from ``operator-assistance-streaming/streaming/multiplex.py``, which is being absorbed.

The hazard, stated rather than hidden
-------------------------------------
Backpressure is honoured per sink, so **the slowest sink paces every other one**. That is the
right default — nothing is silently lost — and it is a real trap: attach a display that stalls
and the transcription stalls with it, which looks like a slow model and is not. The original
carried that caveat in a docstring; here it is also a parameter, because the two cases want
opposite answers.

A stage whose output must be complete (transcription, a stored record) is **lossless**: it has
to wait. A sink whose only job is to show the present (a live meter, a preview pane) is better
served by :data:`LOSSY`, where a full queue drops the oldest frame rather than holding up the
session. Choosing wrong is not a crash, it is a session that feels broken for a reason nobody
can see — which is why the choice is explicit and has no silent default.

Measuring it
------------
``on_wait`` reports how long each ``put`` blocked, per sink. That number is the difference
between "the model is slow" and "a consumer is starving everyone", and it is what
:meth:`vocal_helper.live_metrics.LiveMetrics.record_queue_wait` was built to receive.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")

#: Sink policy: wait for room. Nothing is lost, and this sink can pace the whole fan-out.
LOSSLESS = "lossless"

#: Sink policy: when full, drop the OLDEST pending item to make room. For consumers that show
#: the present rather than record it — a stale frame on a live meter is worth less than a
#: stalled session, and dropping the oldest keeps what is displayed as current as possible.
LOSSY = "lossy"


class Sink:
    """One destination of a fan-out: a queue plus what to do when it is full.

    Parameters
    ----------
    queue : asyncio.Queue
        Where items go. Its ``maxsize`` is what makes the policy matter at all; an unbounded
        queue never blocks and never drops, it just grows until the process dies.
    policy : str, optional
        :data:`LOSSLESS` (default) or :data:`LOSSY`.
    name : str, optional
        Used in ``on_wait`` reports, so a queue-wait figure can be attributed to a consumer
        rather than to "somewhere in the pipeline".
    """

    def __init__(
        self, queue: asyncio.Queue[T | None], *, policy: str = LOSSLESS, name: str = ""
    ) -> None:
        if policy not in (LOSSLESS, LOSSY):
            raise ValueError(f"unknown sink policy {policy!r}; expected {LOSSLESS} or {LOSSY}")
        self.queue = queue
        self.policy = policy
        self.name = name or f"sink{id(self) % 1000}"
        #: Items dropped because this sink was full. Non-zero is information, not an error —
        #: it says a lossy consumer could not keep up, which is exactly what it opted into.
        self.dropped = 0


async def fanout(
    source: asyncio.Queue[T | None],
    sinks: Sequence[Sink | asyncio.Queue[T | None]],
    *,
    on_wait: Callable[[str, float], None] | None = None,
) -> None:
    """Pull from *source* and forward every item to every sink until the stream ends.

    The terminating ``None`` sentinel always reaches every sink, whatever its policy: losing
    an end-of-stream marker is not a late frame, it is a consumer that waits forever. On a
    lossy sink it makes room for itself the same way data does, rather than blocking on a
    queue nobody is draining — the first version awaited it unconditionally and deadlocked.
    On cancellation the sentinel is pushed too, so no downstream stage is left hanging on a
    session that was torn down.

    Parameters
    ----------
    source : asyncio.Queue
        The producer's queue. A bare queue in ``sinks`` is wrapped as a lossless :class:`Sink`,
        so the simple call stays simple.
    on_wait : callable, optional
        ``(sink_name, seconds)`` for every put that had to wait. Feed it to
        :meth:`LiveMetrics.record_queue_wait` to find out which consumer is pacing the rest.
    """
    wrapped = [s if isinstance(s, Sink) else Sink(s) for s in sinks]
    try:
        while True:
            item = await source.get()
            for sink in wrapped:
                await _put(sink, item, on_wait)
            if item is None:
                return
    except asyncio.CancelledError:
        # Tear-down: every consumer still gets its end-of-stream, or it waits forever on a
        # session that no longer exists. Nothing here may await — we are unwinding a
        # cancellation — so the sentinel makes room for itself instead of blocking.
        for sink in wrapped:
            _force_sentinel(sink)
        raise


def _force_sentinel(sink: Sink) -> None:
    """Put the end-of-stream marker without ever blocking and without ever losing it.

    Both callers need exactly this and neither may wait: the lossy branch would hang on a
    queue nobody drains, and the cancellation handler is unwinding and cannot await at all.
    So room is made the way the lossy policy makes room for data — drop the oldest pending
    item and retry — until the sentinel is in.

    This applies to a LOSSLESS sink too, during tear-down only. Discarding a frame from a
    session that is being destroyed costs nothing; discarding its end-of-stream leaves the
    consumer waiting forever, which is the failure this whole function exists to prevent.
    """
    while True:
        try:
            sink.queue.put_nowait(None)
            return
        except asyncio.QueueFull:
            try:
                sink.queue.get_nowait()
                sink.dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - drained concurrently
                pass


async def _put(sink: Sink, item, on_wait: Callable[[str, float], None] | None) -> None:
    """Deliver one item under the sink's policy, timing the wait when anyone is listening."""
    if item is None and sink.policy == LOSSY:
        # The sentinel is never DROPPED, but on a lossy sink it must not block either: a full
        # queue that nobody is draining would hang the tear-down forever. Make room the same
        # way the policy makes room for data — this deadlocked the first version, and the
        # test that caught it is the one named for it.
        _force_sentinel(sink)
        return

    if item is None or sink.policy == LOSSLESS:
        if on_wait is None:
            await sink.queue.put(item)
            return
        t0 = time.monotonic()
        await sink.queue.put(item)
        waited = time.monotonic() - t0
        if waited > 0:
            on_wait(sink.name, waited)
        return

    # Lossy: never block the producer. A full queue loses its oldest pending item, which is
    # the stalest thing a "show the present" consumer holds.
    try:
        sink.queue.put_nowait(item)
    except asyncio.QueueFull:
        try:
            sink.queue.get_nowait()
            sink.dropped += 1
        except asyncio.QueueEmpty:  # pragma: no cover - drained between the two calls
            pass
        try:
            sink.queue.put_nowait(item)
        except asyncio.QueueFull:  # pragma: no cover - another producer refilled it
            sink.dropped += 1
