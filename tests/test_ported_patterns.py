"""LiveKit-inspired eot_bench + Pipecat-inspired parallel_pipelines tests.

Both modules are ported verbatim from pdbms.utils.*. These tests
verify the vocal-helper side stays in sync — same public surface,
same behavioural contracts.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from vocal_helper.eot_bench import (
    DEFAULT_LATENCY_BANDS_MS,
    EOTPair,
    false_cutoff_rate,
    hang_rate,
    median_latency_s,
    score,
)
from vocal_helper.parallel_pipelines import run_parallel_async, run_parallel_sync

# ---------------------------------------------------------------------------
# eot_bench
# ---------------------------------------------------------------------------


def test_eot_pair_is_frozen() -> None:
    """Frozen dataclasses raise ``FrozenInstanceError`` (a ``dataclasses``
    subclass of ``AttributeError``) on assignment ; assert on the base
    ``AttributeError`` so it works across dataclasses/attrs backends."""
    p = EOTPair(true_turn_end_s=1.0, detector_commit_s=1.2)
    with pytest.raises(AttributeError):
        p.true_turn_end_s = 5.0  # type: ignore[misc]


def test_false_cutoff_and_hang_rates() -> None:
    """False-cutoff and hang rates count early / over-budget commits correctly."""
    pairs = [
        EOTPair(10.0, 10.25),  # 250 ms late — OK
        EOTPair(20.0, 20.30),  # 300 ms late — OK
        EOTPair(30.0, 29.80),  # 200 ms EARLY — false cutoff
        EOTPair(40.0, 41.50),  # 1500 ms late — hangs at 300 ms
    ]
    # 1 of 4 is early ≥ 50 ms.
    assert false_cutoff_rate(pairs, tolerance_s=0.05) == pytest.approx(0.25)
    # 2 of 4 are > 300 ms late.
    assert hang_rate(pairs, latency_budget_s=0.300) == pytest.approx(0.5)
    # 1 of 4 is > 1200 ms late.
    assert hang_rate(pairs, latency_budget_s=1.200) == pytest.approx(0.25)


def test_median_latency_and_full_score() -> None:
    """``median_latency_s`` and ``score`` report the expected aggregate shape."""
    pairs = [
        EOTPair(0.0, 0.10),
        EOTPair(0.0, 0.30),
        EOTPair(0.0, 0.50),
    ]
    assert median_latency_s(pairs) == pytest.approx(0.30)
    r = score(pairs)
    assert r["n"] == 3
    assert r["median_latency_ms"] == pytest.approx(300.0)
    assert set(r["hang_rate_at_ms"].keys()) == set(DEFAULT_LATENCY_BANDS_MS)


def test_empty_input_is_safe() -> None:
    """Every metric returns a benign zero on an empty pair list (no ZeroDivision)."""
    assert false_cutoff_rate([], tolerance_s=0.1) == 0.0
    assert hang_rate([], latency_budget_s=0.1) == 0.0
    assert median_latency_s([]) == 0.0
    r = score([])
    assert r["n"] == 0
    assert r["false_cutoff_rate"] == 0.0


def test_negative_tolerance_rejected() -> None:
    """Negative tolerance / budget arguments are rejected with ``ValueError``."""
    with pytest.raises(ValueError):
        false_cutoff_rate([EOTPair(0.0, 0.0)], tolerance_s=-1.0)
    with pytest.raises(ValueError):
        hang_rate([EOTPair(0.0, 0.0)], latency_budget_s=-1.0)


# ---------------------------------------------------------------------------
# parallel_pipelines
# ---------------------------------------------------------------------------


def test_parallel_sync_runs_branches_concurrently() -> None:
    """``run_parallel_sync`` runs branches in threads and keeps input order."""

    def slow_add(x):
        """Branch : sleep 50 ms then return ``x + 1``."""
        time.sleep(0.05)
        return x + 1

    def slow_mul(x):
        """Branch : sleep 50 ms then return ``x * 2``."""
        time.sleep(0.05)
        return x * 2

    def slow_pow(x):
        """Branch : sleep 50 ms then return ``x ** 2``."""
        time.sleep(0.05)
        return x**2

    t0 = time.perf_counter()
    results = run_parallel_sync(
        10,
        [("add", slow_add), ("mul", slow_mul), ("pow", slow_pow)],
    )
    elapsed = time.perf_counter() - t0

    # Values.
    assert results["add"][0] == 11
    assert results["mul"][0] == 20
    assert results["pow"][0] == 100
    # Timing : concurrent execution completes well under the sum of
    # the 3 × 50 ms sleeps.
    assert elapsed < 0.12
    # Ordering : dict preserves input order.
    assert list(results.keys()) == ["add", "mul", "pow"]


def test_parallel_sync_empty_branches_returns_empty() -> None:
    """No branches → empty dict, not an error."""
    assert run_parallel_sync(42, []) == {}


def test_parallel_async_runs_branches_concurrently() -> None:
    """``run_parallel_async`` awaits branches concurrently on one event loop."""

    async def slow_add(x):
        """Async branch : await 50 ms then return ``x + 1``."""
        await asyncio.sleep(0.05)
        return x + 1

    async def slow_mul(x):
        """Async branch : await 50 ms then return ``x * 2``."""
        await asyncio.sleep(0.05)
        return x * 2

    async def main():
        """Run both async branches and assert values + concurrent timing."""
        t0 = time.perf_counter()
        results = await run_parallel_async(
            10,
            [("add", slow_add), ("mul", slow_mul)],
        )
        elapsed = time.perf_counter() - t0
        assert results["add"][0] == 11
        assert results["mul"][0] == 20
        assert elapsed < 0.10  # both 50 ms sleeps run concurrently

    asyncio.run(main())


def test_parallel_async_empty_branches_returns_empty() -> None:
    """No async branches → empty dict, not an error."""

    async def main():
        """Await the empty-branch call so we can assert on its result."""
        return await run_parallel_async(1, [])

    assert asyncio.run(main()) == {}
