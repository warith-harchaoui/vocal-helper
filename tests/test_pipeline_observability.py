"""Regression tests for the pipeline's exception observability.

Before the DiarizeOutput incident the pipeline swallowed *every*
exception in tees and task cleanup via ``contextlib.suppress(Exception)``.
That let a live-in-flight ``AttributeError`` from pyannote 3.x deadlock
the whole test suite for six minutes with zero user-facing signal.

These tests pin the new contract :

- A subscriber that raises must NOT break the pipeline (unchanged).
- A subscriber that raises MUST leave a WARNING record on the
  ``vocal_helper.pipeline`` logger (new).
- A shutdown task that crashes with a non-``CancelledError`` must
  also produce a WARNING record (new).

We don't spin up a real Pipeline here — we call the module-level
helpers directly, which is the whole point of factoring them out.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from vocal_helper.pipeline import (
    _await_task_swallow,
    _invoke_subscribers,
)

# ---------------------------------------------------------------------------
# _invoke_subscribers
# ---------------------------------------------------------------------------


def test_invoke_subscribers_swallows_and_continues() -> None:
    """A raising subscriber does not stop later ones from firing."""
    seen: list[str] = []

    async def bad(_item: object) -> None:
        """First subscriber : always raises to exercise the swallow path."""
        raise ValueError("boom")

    async def good(_item: object) -> None:
        """Second subscriber : records that it ran despite the first raising."""
        seen.append("ran")

    # ``bad`` raising must not short-circuit the list — ``good`` still fires.
    asyncio.run(_invoke_subscribers([bad, good], object(), "test"))
    assert seen == ["ran"], "second subscriber didn't run after first raised"


def test_invoke_subscribers_logs_warning_with_stage_and_callback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failure must be observable via the pipeline logger."""

    async def kaboom(_item: object) -> None:
        """A subscriber that raises with a distinctive message we can grep for."""
        raise RuntimeError("kaboom")

    # Capture at WARNING on the pipeline logger — the contract says the swallow
    # must still leave a record here, tagged with the stage name ("diar").
    with caplog.at_level(logging.WARNING, logger="vocal_helper.pipeline"):
        asyncio.run(_invoke_subscribers([kaboom], object(), "diar"))

    records = [r for r in caplog.records if r.name == "vocal_helper.pipeline"]
    assert records, "no WARNING record produced"
    msg = records[0].getMessage()
    assert "kaboom" in str(records[0].exc_info[1]), (
        "traceback did not surface the underlying exception"
    )
    assert "diar" in msg, "stage name not in the log message"
    assert "kaboom" in msg or "test_invoke_subscribers_logs_warning" in msg, (
        "callback identity not surfaced"
    )


def test_invoke_subscribers_empty_list_is_a_noop() -> None:
    """No subscribers → no logs, no errors."""
    asyncio.run(_invoke_subscribers([], object(), "test"))


# ---------------------------------------------------------------------------
# _await_task_swallow
# ---------------------------------------------------------------------------


def test_await_task_swallow_cancelled_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CancelledError is the expected shutdown path — must NOT log."""

    async def sleeper() -> None:
        await asyncio.sleep(10)  # will be cancelled before firing

    async def drive() -> None:
        t = asyncio.create_task(sleeper(), name="voh.test.sleeper")
        t.cancel()
        await _await_task_swallow(t)

    with caplog.at_level(logging.WARNING, logger="vocal_helper.pipeline"):
        asyncio.run(drive())

    warning_records = [
        r
        for r in caplog.records
        if r.name == "vocal_helper.pipeline" and r.levelno >= logging.WARNING
    ]
    assert not warning_records, f"CancelledError should not log a warning, got: {warning_records}"


def test_await_task_swallow_other_exception_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stage that crashes with a real exception MUST leave a trace."""

    async def crasher() -> None:
        raise AttributeError("'DiarizeOutput' object has no attribute 'itertracks'")

    async def drive() -> None:
        t = asyncio.create_task(crasher(), name="voh.test.diar")
        await _await_task_swallow(t)

    with caplog.at_level(logging.WARNING, logger="vocal_helper.pipeline"):
        asyncio.run(drive())

    records = [
        r
        for r in caplog.records
        if r.name == "vocal_helper.pipeline" and r.levelno >= logging.WARNING
    ]
    assert records, "no WARNING record produced for crashed task"
    # The DiarizeOutput regression is the poster-child ; make estate that
    # exact class of failure now leaves a full traceback.
    assert records[0].exc_info is not None, "no traceback attached"
    assert "DiarizeOutput" in str(records[0].exc_info[1]), (
        "exception message did not reach the log record"
    )
    # Task name should be in the log message.
    assert "voh.test.diar" in records[0].getMessage(), (
        "task name not surfaced ; harder to debug in a multi-stage pipeline"
    )
