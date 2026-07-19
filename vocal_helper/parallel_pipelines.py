"""
vocal_helper.parallel_pipelines
===============================

Fan-out primitive for research A/B comparisons — imports the
Pipecat ``ParallelPipelines`` pattern
(https://docs.pipecat.ai/guides/learn/pipeline) into pdbms.

Motivation
----------
Every diar / STT / EOT study we run has the same shape :

    load audio → run VAD → for each backend branch → collect results

The naive script re-loads the audio and re-runs the VAD once per
branch. That's wasteful (VAD is fast but not free ; a 30-min
meeting × 5 backends × 3 re-runs adds up on the multi-hour
cascades). Pipecat's ``ParallelPipelines`` solves this by carrying
one upstream produce through multiple downstream consumers, with
an ordering contract on the outputs.

This module gives the same primitive to pdbms study scripts. It's
intentionally small — the goal is a clean pattern, not a full
framework.

Two shapes are supported :

1. :func:`run_parallel_sync` — synchronous. The caller supplies a
   list of ``(name, callable)`` branches. Each callable takes the
   shared upstream input and returns a per-branch result. They run
   in parallel via :mod:`concurrent.futures`. Best for CPU-bound
   backends that don't share GPU state.

2. :func:`run_parallel_async` — asyncio-based. The caller supplies
   coroutines. Each coroutine takes the shared input and returns a
   result. They run concurrently via :func:`asyncio.gather`. Best
   for I/O-bound backends (Ollama HTTP, network models).

Output contract : both variants return a dict
``{branch_name: (result, wall_time_s)}`` preserving the order of
the input branch list, so downstream summary tables compare like
for like.

Example
-------

::

    from pdbms.utils.parallel_pipelines import run_parallel_sync

    def run_pyannote(audio): ...
    def run_titanet(audio): ...

    results = run_parallel_sync(
        upstream_input=audio_pcm,
        branches=[
            ("pyannote", run_pyannote),
            ("titanet",  run_titanet),
        ],
    )

    for name, (segs, wall) in results.items():
        print(name, len(segs), wall)

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel_sync(
    upstream_input: T,
    branches: list[tuple[str, Callable[[T], R]]],
    *,
    max_workers: int | None = None,
) -> dict[str, tuple[R, float]]:
    """Run every branch on the same upstream input in parallel.

    Parameters
    ----------
    upstream_input
        The shared value every branch consumes — typically an audio
        buffer, a segment list, or a DiarSegment stream. Branches
        must not mutate it (defensive copy inside each branch is
        the caller's responsibility).
    branches
        Ordered list of ``(name, callable)`` pairs. The order in the
        returned dict matches the input order (Python 3.7+ dict
        insertion order is guaranteed).
    max_workers
        Concurrency cap. Defaults to ``len(branches)`` which is
        usually what you want for a compare-N-backends sweep.

    Returns
    -------
    dict
        ``{name: (result, wall_seconds)}`` in the input order.
    """
    # Empty sweep — nothing to fan out over, return an empty table.
    if not branches:
        return {}
    # One worker per branch by default : a compare-N-backends sweep wants
    # all branches running at once, not queued behind a small pool.
    max_workers = max_workers or len(branches)
    results: dict[str, tuple[R, float]] = {}

    def _time_one(name: str, fn: Callable[[T], R]) -> tuple[str, R, float]:
        """Run one branch on the shared input and wall-time it."""
        # ``perf_counter`` (monotonic, high-res) is the right clock for a
        # duration measurement — immune to wall-clock adjustments.
        t0 = time.perf_counter()
        out = fn(upstream_input)
        return name, out, time.perf_counter() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Submit every branch ; they run concurrently on the pool threads.
        futures = [pool.submit(_time_one, name, fn) for name, fn in branches]
        # Branches finish in completion order, but the caller expects the
        # ORIGINAL branch order — remember each name's slot and reassemble.
        order = {name: i for i, (name, _) in enumerate(branches)}
        rows: list[tuple[str, R, float] | None] = [None] * len(branches)
        # Drain as results land (fail-fast on the first exception raised).
        for f in concurrent.futures.as_completed(futures):
            name, out, wall = f.result()
            rows[order[name]] = (name, out, wall)
    # Rebuild the dict in input order — insertion order is the public contract.
    for name, out, wall in rows:  # type: ignore[misc]
        results[name] = (out, wall)
    return results


async def run_parallel_async(
    upstream_input: T,
    branches: list[tuple[str, Callable[[T], Awaitable[R]]]],
) -> dict[str, tuple[R, float]]:
    """Run every async branch on the same upstream input concurrently.

    Same shape as :func:`run_parallel_sync` but for coroutines.
    Uses :func:`asyncio.gather` so a failure in one branch cancels
    the others ; wrap in try/except at the branch level if
    per-branch fault isolation matters.
    """
    # Empty sweep — nothing to await, return an empty table.
    if not branches:
        return {}

    async def _time_one(name: str, fn: Callable[[T], Awaitable[R]]) -> tuple[str, R, float]:
        """Await one branch coroutine on the shared input and wall-time it."""
        # Same monotonic clock as the sync path so the two variants report
        # wall-times on the identical scale.
        t0 = time.perf_counter()
        out = await fn(upstream_input)
        return name, out, time.perf_counter() - t0

    # ``gather`` preserves argument order in its result list regardless of
    # which coroutine finishes first — so the dict comes out in branch order
    # without any explicit re-sort (unlike the thread pool above).
    triplets = await asyncio.gather(*(_time_one(name, fn) for name, fn in branches))
    return {name: (out, wall) for name, out, wall in triplets}
