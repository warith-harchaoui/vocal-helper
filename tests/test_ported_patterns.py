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


def test_eot_pair_frozen_and_commit_rates() -> None:
    """``EOTPair`` is immutable and the commit-timing rates count correctly.

    Two contracts in one flow. First, ``EOTPair`` is a frozen dataclass, so
    assignment raises ``AttributeError`` (``FrozenInstanceError`` is a
    subclass — asserting on the base keeps this backend-agnostic across
    dataclasses/attrs). Second, over a fixed set of commits,
    ``false_cutoff_rate`` counts the early ones and ``hang_rate`` counts the
    over-budget ones, with the budget threshold shifting the hang count.
    """
    # Immutability: reassigning a field on a frozen instance must raise.
    p = EOTPair(true_turn_end_s=1.0, detector_commit_s=1.2)
    with pytest.raises(AttributeError):
        p.true_turn_end_s = 5.0  # type: ignore[misc]

    pairs = [
        EOTPair(10.0, 10.25),  # 250 ms late — OK
        EOTPair(20.0, 20.30),  # 300 ms late — OK
        EOTPair(30.0, 29.80),  # 200 ms EARLY — false cutoff
        EOTPair(40.0, 41.50),  # 1500 ms late — hangs at 300 ms
    ]
    # 1 of 4 commits ≥ 50 ms early.
    assert false_cutoff_rate(pairs, tolerance_s=0.05) == pytest.approx(0.25)
    # 2 of 4 land > 300 ms late; only 1 lands > 1200 ms late — budget shifts count.
    assert hang_rate(pairs, latency_budget_s=0.300) == pytest.approx(0.5)
    assert hang_rate(pairs, latency_budget_s=1.200) == pytest.approx(0.25)


def test_eot_median_latency_and_full_score() -> None:
    """``median_latency_s`` and ``score`` report the expected aggregate shape.

    ``median_latency_s`` returns the middle commit lag, and ``score`` rolls
    the whole pair list into a report dict whose ``n``, ``median_latency_ms``
    and per-band ``hang_rate_at_ms`` keys match the default latency bands.
    """
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


def test_eot_edge_cases_empty_and_negative_args() -> None:
    """Empty input is benign ; negative tolerance / budget is rejected.

    On an empty pair list every metric returns a benign zero (no
    ``ZeroDivisionError``) and ``score`` reports ``n == 0``. Negative
    ``tolerance_s`` / ``latency_budget_s`` arguments are nonsensical and must
    raise ``ValueError`` rather than silently mis-count.
    """
    # Empty input: no division blow-up, zeros throughout.
    assert false_cutoff_rate([], tolerance_s=0.1) == 0.0
    assert hang_rate([], latency_budget_s=0.1) == 0.0
    assert median_latency_s([]) == 0.0
    r = score([])
    assert r["n"] == 0
    assert r["false_cutoff_rate"] == 0.0

    # Negative thresholds are invalid arguments, not edge data.
    with pytest.raises(ValueError):
        false_cutoff_rate([EOTPair(0.0, 0.0)], tolerance_s=-1.0)
    with pytest.raises(ValueError):
        hang_rate([EOTPair(0.0, 0.0)], latency_budget_s=-1.0)


# ---------------------------------------------------------------------------
# parallel_pipelines
# ---------------------------------------------------------------------------


def test_parallel_sync_concurrency_values_order_and_empty() -> None:
    """``run_parallel_sync`` runs threads concurrently, keeps order, and no-ops on empty.

    Three branches each sleep 200 ms; serial execution would take 600 ms,
    concurrent execution ~200 ms plus thread-pool overhead. The sleep is
    deliberately long so the concurrency gap dwarfs scheduler jitter (a
    shorter sleep flaked on loaded CI). We assert the per-branch return
    values, that the result dict preserves input order, that wall-clock time
    stays well under the serial sum, and that an empty branch list returns an
    empty dict rather than erroring.
    """

    def slow_add(x):
        """Branch : sleep 200 ms then return ``x + 1``."""
        time.sleep(0.2)
        return x + 1

    def slow_mul(x):
        """Branch : sleep 200 ms then return ``x * 2``."""
        time.sleep(0.2)
        return x * 2

    def slow_pow(x):
        """Branch : sleep 200 ms then return ``x ** 2``."""
        time.sleep(0.2)
        return x**2

    t0 = time.perf_counter()
    results = run_parallel_sync(
        10,
        [("add", slow_add), ("mul", slow_mul), ("pow", slow_pow)],
    )
    elapsed = time.perf_counter() - t0

    # Values: each branch received the same input and ran to completion.
    assert results["add"][0] == 11
    assert results["mul"][0] == 20
    assert results["pow"][0] == 100
    # Timing: ~200 ms + overhead proves overlap; the 450 ms bound leaves
    # generous headroom below the 600 ms serial sum for CI jitter.
    assert elapsed < 0.45
    # Ordering: the dict preserves branch-declaration order.
    assert list(results.keys()) == ["add", "mul", "pow"]

    # No branches → empty dict, not an error.
    assert run_parallel_sync(42, []) == {}


def test_parallel_async_concurrency_values_and_empty() -> None:
    """``run_parallel_async`` awaits branches concurrently and no-ops on empty.

    Two async branches each await 50 ms on one event loop; run concurrently
    the whole call finishes under 100 ms (not the 100 ms serial sum). We
    assert the per-branch values, the concurrent timing, and that an empty
    branch list awaits to an empty dict.
    """

    async def slow_add(x):
        """Async branch : await 50 ms then return ``x + 1``."""
        await asyncio.sleep(0.05)
        return x + 1

    async def slow_mul(x):
        """Async branch : await 50 ms then return ``x * 2``."""
        await asyncio.sleep(0.05)
        return x * 2

    async def main() -> None:
        """Drive both branches + the empty case and assert on each."""
        t0 = time.perf_counter()
        results = await run_parallel_async(
            10,
            [("add", slow_add), ("mul", slow_mul)],
        )
        elapsed = time.perf_counter() - t0
        assert results["add"][0] == 11
        assert results["mul"][0] == 20
        assert elapsed < 0.10  # both 50 ms sleeps overlap on one loop

        # No branches → empty dict, not an error.
        assert await run_parallel_async(1, []) == {}

    asyncio.run(main())
