"""
vocal_helper.llm
================

Optional LLM analyst stage. Consumes :class:`Utterance` events and
maintains a **rolling summary** of the conversation up to
``recent_window_s`` (default 60 s) before now.

Algorithm
---------
- Keep a deque of recent utterances with timestamps.
- After every new :class:`Utterance` :
  - Move utterances whose ``t1`` is older than
    ``now − recent_window_s`` from the recent buffer into the
    summarisation queue.
  - If the summarisation queue grew by ``flush_every_n`` (default 5)
    new utterances, ask the LLM to fold them into the running
    ``summary`` field.
  - Emit a :class:`SummarySnapshot` with the current
    ``(summary, recent)`` pair.

The model is never hard-coded here. The stage receives a resolved
**engine descriptor** (from ``best_engine_ai_helper.ensure`` on the
package's ``llm.brief.yaml``) and routes every request through
``best_engine_ai_helper.llm.chat`` — which dispatches to Ollama or vLLM
per the descriptor. ``chat`` is synchronous, so the call is offloaded to
a worker thread to keep the event loop responsive.

If the LLM is unreachable, the stage logs a warning and emits a
:class:`SummarySnapshot` with the previous ``summary`` unchanged,
so downstream consumers never miss an event.

Author
------
Warith HARCHAOUI , https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import os_helper as osh
from best_engine_ai_helper import llm

from vocal_helper.types import SummarySnapshot, Utterance

DEFAULT_RECENT_WINDOW_S = 60.0
# ``flush_every_n`` is the count-based fallback — refresh the summary
# every N evicted utterances. Used only when ``flush_every_s`` is
# explicitly set to ``None`` by the caller.
DEFAULT_FLUSH_EVERY_N = 5
# ``flush_every_s`` is the canonical time-based cadence — refresh the
# summary whenever the accumulated evicted-content duration crosses
# this many seconds.
#
# Default 60.0 — selected from two complementary 2026-06-30 sweeps :
#
# ``studies/llm_cadence_sweep.py`` — single-meeting (AMI IS1008a) :
#   config     RTF    cos_sim
#   n=20      0.260    0.397
#   t=30s     0.490    0.414
#   t=60s     0.311    0.420   ← highest cos_sim
#   t=120s    0.192    0.407
#
# ``studies/llm_cadence_sweep_multi.py`` — pooled median across
# 4 AMI meetings (IS1008a + ES2011a + ES2011d + TS3004a) :
#   config    med_RTF  med_n  med_cos
#   n=20       0.369    23    0.354  ← highest med_cos
#   t=60s      0.278    17    0.339
#   t=120s    0.181     9    0.315
#
# The multi-meeting median crowns ``n=20`` on cos_sim but t=60s
# remains the production pick :
#   - the (0.354 − 0.339) cos_sim gap is well within the inter-meeting
#     noise (cos_sim ranges from 0.279 to 0.471 for the same config) ;
#   - t=60s is 25 % faster (RTF 0.278 vs 0.369) ;
#   - time-based cadence delivers a predictable "summary refreshes
#     every ~ 1 minute" UX regardless of how chatty the speakers are ;
#   - matches the user spec "rolling summary up to 1 minute before now".
DEFAULT_FLUSH_EVERY_S: float | None = 60.0
DEFAULT_SUMMARY_PROMPT = (
    "You are a meeting note-taker. Update the running summary below "
    "by integrating the new utterances. Keep it concise (≤ 6 bullet "
    "points), preserve speaker attributions, and drop low-signal "
    "small talk. Output only the updated summary, nothing else.\n\n"
    "Current summary:\n{summary}\n\n"
    "New utterances (older → newer):\n{new_block}\n"
)


@dataclass
class _Buffer:
    """In-memory state for the analyst."""

    summary: str = ""
    recent: deque[Utterance] = field(default_factory=deque)
    pending_for_summary: list[Utterance] = field(default_factory=list)


class GemmaAnalystStage:
    """Producer/consumer LLM analyst with a rolling summary.

    Parameters
    ----------
    engine : dict
        Resolved engine descriptor from
        ``best_engine_ai_helper.ensure(<vocal_helper package dir>)`` —
        it names the backend (Ollama / vLLM), the base URL, and the
        text model to serve. No default model is baked in here.
    recent_window_s : float
        How many seconds of verbatim transcript to keep before
        folding into the summary. Default 60 s.
    flush_every_n : int
        Update the summary every ``flush_every_n`` new utterances
        that crossed the recent window. Default 5.
    prompt_template : str
        Override the canonical summarisation prompt. Two
        placeholders : ``{summary}`` (current digest) and
        ``{new_block}`` (newly evicted utterances). Default keeps a
        ≤ 6-bullet meeting digest.
    """

    def __init__(
        self,
        *,
        engine: dict[str, Any],
        recent_window_s: float = DEFAULT_RECENT_WINDOW_S,
        flush_every_n: int = DEFAULT_FLUSH_EVERY_N,
        flush_every_s: float | None = DEFAULT_FLUSH_EVERY_S,
        prompt_template: str = DEFAULT_SUMMARY_PROMPT,
    ) -> None:
        """Configure the analyst stage without touching the network.

        Nothing here reaches the model — ``engine`` is a plain descriptor and
        every request is issued later through :func:`llm.chat`. Constructing
        the stage stays cheap and import-safe.

        Parameters
        ----------
        engine : dict
            Resolved engine descriptor (see the class docstring). The model
            tag surfaced on each :class:`SummarySnapshot` is read from it.
        recent_window_s : float
            Utterances newer than this many seconds stay in the verbatim
            ``recent`` buffer ; older ones are evicted into the summary.
        flush_every_n : int
            Count-based cadence fallback — refresh the summary after this
            many evicted utterances. Used only when ``flush_every_s`` is
            ``None``.
        flush_every_s : float | None
            Duration-based cadence — refresh once the evicted block spans
            this many seconds. Takes precedence over ``flush_every_n``.
        prompt_template : str
            Summarisation prompt with ``{summary}`` / ``{new_block}``
            placeholders.
        """
        self._engine = engine
        # Model tag stamped on each snapshot (read from the resolved engine,
        # never a hard-coded constant). Falls back to the empty string if the
        # descriptor has no llm section.
        self.model = str(engine.get("llm", {}).get("model", "")) if isinstance(engine, dict) else ""
        self.recent_window_s = recent_window_s
        self.flush_every_n = flush_every_n
        self.flush_every_s = flush_every_s
        self.prompt_template = prompt_template
        self._buf = _Buffer()
        # Track the t0 of the oldest pending-for-summary utterance so
        # the time-based cadence can fire on duration accumulated.
        self._oldest_pending_t0: float | None = None

    # ----- public coroutine ----------------------------------------------

    async def run(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
    ) -> None:
        """Consume :class:`Utterance` from ``inbox``, push :class:`SummarySnapshot`."""
        while True:
            item = await inbox.get()
            if item is None:
                # Flush remaining recent items into the summary on shutdown.
                if self._buf.recent:
                    self._buf.pending_for_summary.extend(self._buf.recent)
                    self._buf.recent.clear()
                if self._buf.pending_for_summary:
                    self._buf.summary = await asyncio.to_thread(self._summarise)
                snap = self._snapshot(item_t=None)
                if snap is not None:
                    await outbox.put(snap)
                await outbox.put(None)
                return
            snap = await self._on_utterance(item)
            if snap is not None:
                await outbox.put(snap)

    # ----- core ---------------------------------------------------------

    async def _on_utterance(self, utt: Utterance) -> SummarySnapshot | None:
        """Ingest one utterance and, if cadence fires, refresh the summary.

        Appends ``utt`` to the verbatim recent buffer, evicts anything
        older than ``recent_window_s`` into the pending-for-summary
        queue, then decides whether the accumulated block warrants an
        LLM refresh (duration cadence first, count cadence as fallback).

        Parameters
        ----------
        utt : Utterance
            The newly transcribed utterance ; ``t1`` is treated as
            "now" for windowing.

        Returns
        -------
        SummarySnapshot | None
            The current ``(summary, recent)`` snapshot, or ``None`` for
            an empty (VAD-blip) utterance that carries no text.
        """
        if not utt["text"].strip():
            return None  # empty utterance — VAD blip
        now = utt["t1"]
        self._buf.recent.append(utt)
        while self._buf.recent and (now - self._buf.recent[0]["t1"]) > self.recent_window_s:
            evicted = self._buf.recent.popleft()
            self._buf.pending_for_summary.append(evicted)
            if self._oldest_pending_t0 is None:
                self._oldest_pending_t0 = evicted["t0"]
        # Decide whether to refresh the summary :
        # - ``flush_every_s`` takes precedence when set ;
        # - otherwise fall back to ``flush_every_n``.
        should_flush = False
        if self.flush_every_s is not None and self._buf.pending_for_summary:
            newest_pending_t1 = self._buf.pending_for_summary[-1]["t1"]
            # ``_oldest_pending_t0`` is a real timestamp that can legitimately be
            # ``0.0`` (the very first utterance of a session starts at t=0), so
            # test it with ``is None`` — a truthiness check would treat that
            # cold-start ``0.0`` as "unset" and collapse the span to zero, which
            # would silently disable the time cadence until the queue rolled
            # past the first utterance.
            oldest_t0 = (
                self._oldest_pending_t0
                if self._oldest_pending_t0 is not None
                else newest_pending_t1
            )
            span_s = newest_pending_t1 - oldest_t0
            if span_s >= self.flush_every_s:
                should_flush = True
        elif len(self._buf.pending_for_summary) >= self.flush_every_n:
            should_flush = True
        if should_flush:
            self._buf.summary = await asyncio.to_thread(self._summarise)
            self._oldest_pending_t0 = None
        return self._snapshot(item_t=now)

    def _summarise(self) -> str:
        """Fold the pending block into the running summary via one LLM call.

        ``llm.chat`` is synchronous (blocks on ``requests``), so callers invoke
        this through :func:`asyncio.to_thread` to keep the event loop responsive.

        Returns
        -------
        str
            The refreshed digest. On an empty pending queue the previous
            summary is returned unchanged ; on a network / model error
            the previous summary is kept and the poisoned block dropped
            (so we never retry the same failing batch forever).
        """
        if not self._buf.pending_for_summary:
            return self._buf.summary
        new_block = "\n".join(
            f"[{u['t0']:.1f}-{u['t1']:.1f}] {u['speaker']}: {u['text']}"
            for u in self._buf.pending_for_summary
        )
        prompt = self.prompt_template.format(
            summary=self._buf.summary or "(none yet)",
            new_block=new_block,
        )
        try:
            text = llm.chat(prompt, engine=self._engine, kind="llm")
        except Exception as exc:  # noqa: BLE001
            # Network or model error — keep old summary, drop the block
            # so we don't infinitely retry the same poisoned batch.
            osh.warning(f"GemmaAnalystStage: llm.chat failed ({exc!r}); keeping previous summary")
            self._buf.pending_for_summary.clear()
            return self._buf.summary
        self._buf.pending_for_summary.clear()
        return str(text).strip()

    def _snapshot(self, item_t: float | None) -> SummarySnapshot | None:
        """Build a :class:`SummarySnapshot` from the current buffer state.

        Parameters
        ----------
        item_t : float | None
            Timestamp to stamp the snapshot with. When ``None`` (the
            shutdown flush), we fall back to the last recent utterance's
            ``t1``, or ``0.0`` if the recent buffer is already empty.

        Returns
        -------
        SummarySnapshot | None
            The snapshot, or ``None`` when there is nothing to report
            yet (neither recent utterances nor a running summary) so
            downstream consumers are not woken for an empty event.
        """
        if not self._buf.recent and not self._buf.summary:
            return None
        recent_block = "\n".join(
            f"[{u['t0']:.1f}-{u['t1']:.1f}] {u['speaker']}: {u['text']}" for u in self._buf.recent
        )
        t = (
            item_t
            if item_t is not None
            else (self._buf.recent[-1]["t1"] if self._buf.recent else 0.0)
        )
        return SummarySnapshot(
            t0=t,
            summary=self._buf.summary,
            recent=recent_block,
            model=self.model,
        )
