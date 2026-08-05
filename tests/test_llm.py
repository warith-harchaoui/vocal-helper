"""Tests for the ``vocal_helper.llm`` surface that can run offline.

The analyst no longer speaks to Ollama directly — it routes every request
through ``best_engine_ai_helper.llm.chat`` using a resolved engine descriptor.
These tests never hit the network: they patch ``vocal_helper.llm.llm.chat``
with a recorder that captures the prompt and replays a scripted reply / error.
"""

from __future__ import annotations

import asyncio

import pytest

import vocal_helper.llm as llm_mod
from vocal_helper.llm import (
    DEFAULT_FLUSH_EVERY_N,
    DEFAULT_RECENT_WINDOW_S,
    GemmaAnalystStage,
)
from vocal_helper.types import Utterance

# A resolved engine descriptor, as ``best_engine_ai_helper.ensure`` would hand
# back — the analyst reads ``["llm"]["model"]`` for the snapshot tag and passes
# the whole dict to ``llm.chat``. No network is implied by holding it.
FAKE_ENGINE = {
    "backend": "ollama",
    "base_url": "http://localhost:11434",
    "llm": {"model": "test-model:latest"},
}


def _utt(t0: float, t1: float, text: str, speaker: str = "S0") -> Utterance:
    """Build a minimal :class:`Utterance` for the analyst tests."""
    return Utterance(t0=t0, t1=t1, speaker=speaker, text=text, words=[], language="en")


class _FakeChat:
    """Records prompts and replays a scripted response / error.

    Drop-in for ``best_engine_ai_helper.llm.chat`` — same keyword-only
    ``engine`` / ``kind`` signature. ``prompts`` captures what the stage sent
    (asserting prompt assembly) and an ``error`` makes the call raise
    (exercising the failure path). No network is ever touched.
    """

    def __init__(self, response: str = "digest", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.prompts: list[str] = []
        self.engines: list[object] = []
        self.calls = 0

    def __call__(self, prompt: str, *, engine: object = None, kind: str | None = None) -> str:
        """Mimic ``llm.chat`` — capture the prompt/engine, then reply or raise."""
        self.calls += 1
        self.prompts.append(prompt)
        self.engines.append(engine)
        if self.error is not None:
            raise self.error
        return self.response


def _patch_chat(monkeypatch: pytest.MonkeyPatch, fake: _FakeChat) -> None:
    """Route ``vocal_helper.llm.llm.chat`` through ``fake`` for one test."""
    monkeypatch.setattr(llm_mod.llm, "chat", fake)


# ---------------------------------------------------------------------------
# GemmaAnalystStage — construction only, never connects to a model.
# ---------------------------------------------------------------------------


def test_analyst_defaults() -> None:
    """Bare ``GemmaAnalystStage`` mirrors the documented defaults; model read from engine."""
    stage = GemmaAnalystStage(engine=FAKE_ENGINE)
    assert stage._engine is FAKE_ENGINE
    # The snapshot tag comes from the resolved engine, never a hard-coded constant.
    assert stage.model == "test-model:latest"
    assert stage.recent_window_s == DEFAULT_RECENT_WINDOW_S
    assert stage.flush_every_n == DEFAULT_FLUSH_EVERY_N
    # Default 60.0 (time-based) selected in the 2026-06-30 cadence
    # sweep — see vocal_helper.llm module docstring for the table.
    assert stage.flush_every_s == 60.0


def test_analyst_accepts_overrides() -> None:
    """Every constructor override is stored verbatim on the analyst stage."""
    stage = GemmaAnalystStage(
        engine=FAKE_ENGINE,
        recent_window_s=30.0,
        flush_every_n=10,
        flush_every_s=15.0,
        prompt_template="custom {summary} {new_block}",
    )
    assert stage.recent_window_s == 30.0
    assert stage.flush_every_n == 10
    assert stage.flush_every_s == 15.0
    assert "{summary}" in stage.prompt_template


# ---------------------------------------------------------------------------
# _summarise — prompt assembly, empty-queue short-circuit, LLM failure path.
# The chat call is a fake ; the network is never touched.
# ---------------------------------------------------------------------------


def test_summarise_assembles_prompt_and_folds_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pending block is rendered into the prompt and replaced by the LLM digest.

    Pins the summarisation contract end-to-end (minus the network): the evicted
    utterances are formatted as timestamped speaker lines, the current summary
    fills the ``{summary}`` slot, the reply becomes the new summary, and the
    pending queue is cleared so the same block is never re-summarised. The engine
    descriptor is threaded through to ``llm.chat`` untouched.
    """
    stage = GemmaAnalystStage(engine=FAKE_ENGINE)
    fake = _FakeChat(response="  • Alice proposed a date  ")
    _patch_chat(monkeypatch, fake)
    stage._buf.summary = "prior digest"
    stage._buf.pending_for_summary = [
        _utt(0.0, 2.0, "let's pick a date", "Alice"),
        _utt(2.0, 4.0, "how about Friday", "Bob"),
    ]

    out = stage._summarise()

    assert out == "• Alice proposed a date"  # reply, stripped, becomes summary
    assert stage._buf.pending_for_summary == []  # block consumed, never re-sent
    assert fake.engines[0] is FAKE_ENGINE  # engine forwarded to chat unchanged
    prompt = fake.prompts[0]
    assert "prior digest" in prompt  # {summary} slot filled with the running digest
    assert "Alice: let's pick a date" in prompt  # utterance rendered with speaker
    assert "[0.0-2.0]" in prompt  # timestamp prefix rendered


def test_summarise_empty_queue_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing pending, ``_summarise`` returns the prior summary and never
    calls the LLM (no wasted round-trip on an empty flush)."""
    stage = GemmaAnalystStage(engine=FAKE_ENGINE)
    fake = _FakeChat()
    _patch_chat(monkeypatch, fake)
    stage._buf.summary = "unchanged"
    assert stage._summarise() == "unchanged"
    assert fake.calls == 0  # empty pending → chat never invoked


def test_summarise_keeps_summary_and_drops_block_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed chat call keeps the previous summary and drops the poisoned block.

    The stage must degrade gracefully: on a network/model error it logs a
    warning, preserves the last good summary (downstream never regresses), and
    clears the pending block so the same failing batch is not retried forever.
    """
    stage = GemmaAnalystStage(engine=FAKE_ENGINE)
    _patch_chat(monkeypatch, _FakeChat(error=ConnectionError("engine down")))
    stage._buf.summary = "last good digest"
    stage._buf.pending_for_summary = [_utt(0.0, 2.0, "hello")]

    out = stage._summarise()

    assert out == "last good digest"  # previous summary preserved on failure
    assert stage._buf.pending_for_summary == []  # poisoned block dropped, no retry loop


# ---------------------------------------------------------------------------
# _on_utterance — VAD-blip skip, window eviction, and both flush cadences.
# ---------------------------------------------------------------------------


def test_on_utterance_skips_empty_and_snapshots_recent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only utterance is dropped; a real one lands in the recent
    buffer and is reflected verbatim in the emitted snapshot.

    Guards the VAD-blip guard (empty text → ``None``, nothing buffered) and the
    happy path where a below-window utterance stays verbatim in ``recent`` and no
    LLM flush is triggered.
    """
    fake = _FakeChat()
    _patch_chat(monkeypatch, fake)

    async def drive() -> tuple[object, object]:
        """Feed one blank then one real utterance through ``_on_utterance``."""
        stage = GemmaAnalystStage(engine=FAKE_ENGINE, recent_window_s=60.0)
        blip = await stage._on_utterance(_utt(0.0, 0.5, "   "))
        snap = await stage._on_utterance(_utt(1.0, 3.0, "real words", "Bob"))
        return blip, snap

    blip, snap = asyncio.run(drive())
    assert blip is None  # VAD blip carries no text → no snapshot
    assert snap is not None
    assert "Bob: real words" in snap["recent"]  # verbatim recent transcript
    assert snap["summary"] == ""  # nothing evicted yet → summary untouched
    assert fake.calls == 0  # no flush on a single in-window utterance


def test_time_cadence_evicts_and_flushes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the evicted block spans ``flush_every_s``, exactly one LLM refresh
    fires and the running summary is updated.

    Exercises the duration-based cadence (the production default): utterances
    older than ``recent_window_s`` are evicted, and when the accumulated evicted
    span crosses ``flush_every_s`` a single summarise call folds them into the
    summary. count-based ``flush_every_n`` stays out of the way (set high).
    """
    fake = _FakeChat(response="rolling digest")
    _patch_chat(monkeypatch, fake)

    async def drive() -> object:
        """Push utterances until the evicted span crosses the 10 s cadence."""
        stage = GemmaAnalystStage(
            engine=FAKE_ENGINE,
            recent_window_s=5.0,  # anything older than 5 s gets evicted
            flush_every_s=10.0,  # refresh once the evicted block spans 10 s
            flush_every_n=1000,  # keep the count fallback dormant
        )
        # t1 marches forward; utterances older than 5 s are evicted. Once the
        # evicted span (newest evicted t1 − oldest evicted t0) crosses 10 s the
        # time cadence trips. Timestamps start at 1.0 here to keep this test
        # focused on the general cadence; the cold-start t0==0.0 corner is pinned
        # separately by ``test_time_cadence_fires_from_cold_start_at_t0_zero``.
        for k in range(12):
            t0 = 1.0 + k * 2
            await stage._on_utterance(_utt(t0, t0 + 1.0, f"line {k}"))
        return stage

    stage = asyncio.run(drive())
    assert fake.calls >= 1  # duration cadence fired
    assert stage._buf.summary == "rolling digest"  # summary folded in


def test_time_cadence_fires_from_cold_start_at_t0_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """The time cadence must trip even when the first utterance starts at t0=0.0.

    Regression for a falsy-zero trap in ``_on_utterance``: the oldest-pending
    timestamp was read with ``self._oldest_pending_t0 or newest_pending_t1``,
    which treats a legitimate ``0.0`` (every session's first utterance) as
    "unset" and collapses the evicted span to zero — silently disabling the
    duration cadence from a cold start. The span must be measured from the real
    ``0.0`` origin so the flush fires on schedule.
    """
    fake = _FakeChat(response="cold-start digest")
    _patch_chat(monkeypatch, fake)

    async def drive() -> object:
        """Same cadence as the sibling test, but timestamps begin at t0=0.0."""
        stage = GemmaAnalystStage(
            engine=FAKE_ENGINE,
            recent_window_s=5.0,
            flush_every_s=10.0,
            flush_every_n=1000,  # keep the count fallback dormant
        )
        # First utterance sits at t0=0.0 exactly — the trap the fix addresses.
        for k in range(12):
            t0 = 0.0 + k * 2
            await stage._on_utterance(_utt(t0, t0 + 1.0, f"line {k}"))
        return stage

    stage = asyncio.run(drive())
    assert fake.calls >= 1  # cadence fired despite the t0=0.0 origin
    assert stage._buf.summary == "cold-start digest"


def test_count_cadence_used_when_time_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``flush_every_s=None`` the count fallback flushes every N evictions.

    Pins the documented precedence: the duration cadence is opt-out via ``None``,
    and then the older count-based rule (refresh after ``flush_every_n`` evicted
    utterances) governs instead.
    """
    fake = _FakeChat(response="counted digest")
    _patch_chat(monkeypatch, fake)

    async def drive() -> object:
        """Evict enough utterances to trip the count-based cadence once."""
        stage = GemmaAnalystStage(
            engine=FAKE_ENGINE,
            recent_window_s=1.0,  # evict aggressively so the queue fills fast
            flush_every_s=None,  # disable the time cadence → fall back to count
            flush_every_n=3,  # flush after 3 evicted utterances
        )
        for k in range(6):
            await stage._on_utterance(_utt(float(k * 2), float(k * 2 + 0.5), f"u{k}"))
        return stage

    stage = asyncio.run(drive())
    assert fake.calls >= 1  # count cadence fired
    assert stage._buf.summary == "counted digest"


# ---------------------------------------------------------------------------
# run() — full producer/consumer loop with shutdown flush (mocked chat).
# ---------------------------------------------------------------------------


def test_run_drains_inbox_and_flushes_on_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """The coroutine emits per-utterance snapshots and folds the tail into the
    summary on the ``None`` shutdown sentinel.

    A realistic mini-workflow: two utterances in, then ``None``. We expect a
    snapshot per real utterance, a final summarise of whatever is still in the
    recent buffer at shutdown, and a terminating ``None`` on the outbox — the
    contract downstream stages rely on.
    """
    fake = _FakeChat(response="final digest")
    _patch_chat(monkeypatch, fake)

    async def drive() -> tuple[list, GemmaAnalystStage]:
        """Run the stage against hand-fed inbox/outbox queues."""
        stage = GemmaAnalystStage(engine=FAKE_ENGINE, recent_window_s=60.0)
        inbox: asyncio.Queue = asyncio.Queue()
        outbox: asyncio.Queue = asyncio.Queue()
        await inbox.put(_utt(0.0, 2.0, "opening remark", "Alice"))
        await inbox.put(_utt(2.0, 4.0, "reply", "Bob"))
        await inbox.put(None)  # shutdown sentinel
        await stage.run(inbox, outbox)
        drained = []
        while not outbox.empty():
            drained.append(outbox.get_nowait())
        return drained, stage

    drained, stage = asyncio.run(drive())
    assert drained[-1] is None  # outbox terminated for downstream consumers
    snaps = [s for s in drained if s is not None]
    assert len(snaps) >= 2  # one snapshot per real utterance, plus shutdown flush
    # The shutdown flush folds the still-recent utterances into the summary.
    assert stage._buf.summary == "final digest"
    assert fake.calls == 1  # a single summarise at shutdown
