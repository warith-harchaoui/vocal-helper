"""A run that lost a stage must not report success.

Before StageCrashed existed, a source that raised produced a normal, EMPTY transcript: the
crash went to a log and run() returned cleanly with nothing in it, so a caller saw silence
and no reason for it. That is the failure shape the pipeline's own docstring says once hid a
broken pyannote API "undetected for hours".
"""

import asyncio

import pytest

import vocal_helper as voh
from vocal_helper.pipeline import StageCrashed, _drain_task


def _boom_task() -> asyncio.Task:
    async def boom():
        raise RuntimeError("le moteur est mort")

    return asyncio.ensure_future(boom())


@pytest.mark.asyncio
async def test_a_crash_is_recorded_not_only_logged() -> None:
    """Logging is invisible to a program; recording is what lets run() refuse to lie."""
    failures: list[tuple[str, BaseException]] = []
    await _drain_task(_boom_task(), failures)
    assert len(failures) == 1
    assert isinstance(failures[0][1], RuntimeError)


@pytest.mark.asyncio
async def test_a_cancelled_task_is_not_a_failure() -> None:
    """Cancellation is the normal shutdown path and must stay silent."""

    async def forever():
        await asyncio.sleep(3600)

    t = asyncio.ensure_future(forever())
    await asyncio.sleep(0)
    t.cancel()
    failures: list[tuple[str, BaseException]] = []
    await _drain_task(t, failures)
    assert failures == []


@pytest.mark.asyncio
async def test_draining_continues_past_a_dead_stage() -> None:
    """One dead stage must never leave the others un-awaited."""
    failures: list[tuple[str, BaseException]] = []
    for _ in range(3):
        await _drain_task(_boom_task(), failures)
    assert len(failures) == 3


def test_both_pipelines_are_strict_by_default() -> None:
    assert voh.Pipeline(source=lambda: iter(())).strict is True
    assert voh.OfflinePipeline(source=lambda: iter(())).strict is True


def test_leniency_is_available_but_has_to_be_asked_for() -> None:
    """A choice somebody made on purpose, rather than one inherited by accident."""
    p = voh.Pipeline(source=lambda: iter(()), strict=False)
    assert p.strict is False and p.stage_failures == []


def test_the_error_names_every_stage_that_died() -> None:
    err = StageCrashed([("voh.offline.source", RuntimeError("x")),
                        ("voh.offline.asr", ValueError("y"))])
    assert "voh.offline.source" in str(err) and "voh.offline.asr" in str(err)
    assert "incomplete" in str(err)
    assert len(err.failures) == 2
