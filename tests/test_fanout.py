"""Forking one queue into several: delivery, backpressure, and who gets to stall whom."""

import asyncio

import pytest

from vocal_helper.fanout import LOSSLESS, LOSSY, Sink, fanout


async def _drain(q: asyncio.Queue) -> list:
    out = []
    while True:
        item = await q.get()
        if item is None:
            return out
        out.append(item)


@pytest.mark.asyncio
async def test_every_sink_receives_every_item() -> None:
    src: asyncio.Queue = asyncio.Queue()
    a, b = asyncio.Queue(), asyncio.Queue()
    for x in (1, 2, 3, None):
        src.put_nowait(x)
    await fanout(src, [a, b])
    assert await _drain(a) == [1, 2, 3]
    assert await _drain(b) == [1, 2, 3]


@pytest.mark.asyncio
async def test_a_bare_queue_is_accepted_so_the_simple_call_stays_simple() -> None:
    src: asyncio.Queue = asyncio.Queue()
    only = asyncio.Queue()
    src.put_nowait(None)
    await fanout(src, [only])
    assert only.get_nowait() is None


@pytest.mark.asyncio
async def test_a_slow_lossless_sink_paces_the_producer() -> None:
    """The right default, and a real trap: a stalled display stalls transcription too."""
    src: asyncio.Queue = asyncio.Queue()
    slow = Sink(asyncio.Queue(maxsize=1), name="slow")
    for x in (1, 2, 3, None):
        src.put_nowait(x)

    task = asyncio.ensure_future(fanout(src, [slow]))
    await asyncio.sleep(0.02)
    assert not task.done()              # blocked: the sink is full and nothing was dropped
    assert slow.dropped == 0
    seen = []
    while not task.done() or not slow.queue.empty():
        try:
            seen.append(slow.queue.get_nowait())
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)
    await asyncio.wait_for(task, timeout=2)
    assert [x for x in seen if x is not None] == [1, 2, 3]  # nothing lost, only delayed


@pytest.mark.asyncio
async def test_a_lossy_sink_drops_the_oldest_rather_than_stalling_the_session() -> None:
    """A stale frame on a live meter is worth less than a session that feels broken."""
    src: asyncio.Queue = asyncio.Queue()
    live = Sink(asyncio.Queue(maxsize=2), policy=LOSSY, name="meter")
    for x in (1, 2, 3, 4, None):
        src.put_nowait(x)
    await asyncio.wait_for(fanout(src, [live]), timeout=2)
    # Three dropped, not two: the sentinel also needs room, and it takes it the same way.
    assert live.dropped == 3
    # What survived is the most RECENT item, which is what a "show the present" sink wants,
    # followed by the end-of-stream that always gets through.
    assert live.queue.get_nowait() == 4
    assert live.queue.get_nowait() is None


@pytest.mark.asyncio
async def test_the_end_of_stream_is_lossless_even_for_a_lossy_sink() -> None:
    """Dropping a sentinel is not a late frame, it is a consumer that waits forever."""
    src: asyncio.Queue = asyncio.Queue()
    live = Sink(asyncio.Queue(maxsize=1), policy=LOSSY)
    src.put_nowait(1)
    src.put_nowait(None)
    await asyncio.wait_for(fanout(src, [live]), timeout=2)
    # Room was made for the sentinel rather than blocking on a queue nobody drains.
    assert live.queue.get_nowait() is None


@pytest.mark.asyncio
async def test_cancellation_still_releases_every_consumer() -> None:
    src: asyncio.Queue = asyncio.Queue()
    a = asyncio.Queue()
    task = asyncio.ensure_future(fanout(src, [a]))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert a.get_nowait() is None


@pytest.mark.asyncio
async def test_queue_waits_are_reported_with_the_sink_that_caused_them() -> None:
    """The difference between 'the model is slow' and 'a consumer is starving everyone'."""
    src: asyncio.Queue = asyncio.Queue()
    slow = Sink(asyncio.Queue(maxsize=1), name="asr")
    for x in (1, 2, None):
        src.put_nowait(x)
    waits: list[tuple[str, float]] = []
    task = asyncio.ensure_future(fanout(src, [slow], on_wait=lambda n, s: waits.append((n, s))))
    await asyncio.sleep(0.05)
    while not task.done():
        try:
            slow.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        await asyncio.sleep(0.01)
    await asyncio.wait_for(task, timeout=2)
    assert any(name == "asr" and seconds > 0 for name, seconds in waits)


def test_an_unknown_policy_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown sink policy"):
        Sink(asyncio.Queue(), policy="peut-être")


def test_the_default_policy_loses_nothing() -> None:
    assert Sink(asyncio.Queue()).policy == LOSSLESS
