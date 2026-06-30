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

The LLM call runs in a worker thread (Ollama's HTTP client blocks).
If the LLM is unreachable, the stage logs a warning and emits a
:class:`SummarySnapshot` with the previous ``summary`` unchanged,
so downstream consumers never miss an event.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

from vocal_helper.types import SummarySnapshot, Utterance

DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_RECENT_WINDOW_S = 60.0
DEFAULT_FLUSH_EVERY_N = 5
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
    model : str
        Ollama model tag. Default ``"gemma4:e4b"`` — Gemma 4 4B
        effective, the canonical light analyst across the AI Helpers
        suite. On Apple-Silicon, ``ollama`` resolves the ``-mlx``
        variant automatically when present.
    recent_window_s : float
        How many seconds of verbatim transcript to keep before
        folding into the summary. Default 60 s.
    flush_every_n : int
        Update the summary every ``flush_every_n`` new utterances
        that crossed the recent window. Default 5.
    host : str, optional
        Ollama host URL. Defaults to the ``OLLAMA_HOST`` env var or
        ``http://localhost:11434``.
    prompt_template : str
        Override the canonical summarisation prompt. Two
        placeholders : ``{summary}`` (current digest) and
        ``{new_block}`` (newly evicted utterances). Default keeps a
        ≤ 6-bullet meeting digest.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        recent_window_s: float = DEFAULT_RECENT_WINDOW_S,
        flush_every_n: int = DEFAULT_FLUSH_EVERY_N,
        host: str | None = None,
        prompt_template: str = DEFAULT_SUMMARY_PROMPT,
    ) -> None:
        self.model = model
        self.recent_window_s = recent_window_s
        self.flush_every_n = flush_every_n
        self.host = host
        self.prompt_template = prompt_template
        self._client = None
        self._buf = _Buffer()

    # ----- lifecycle ------------------------------------------------------

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import ollama  # type: ignore
        except ImportError as e:  # noqa: BLE001
            raise ImportError(
                "GemmaAnalystStage requires ollama. "
                "Install with `pip install vocal-helper[llm]`."
            ) from e
        # The newer ollama package exposes a ``Client`` constructor.
        if self.host:
            self._client = ollama.Client(host=self.host)
        else:
            self._client = ollama.Client()

    # ----- public coroutine ----------------------------------------------

    async def run(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
    ) -> None:
        """Consume :class:`Utterance` from ``inbox``, push :class:`SummarySnapshot`."""
        self._ensure_client()
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
        if not utt["text"].strip():
            return None  # empty utterance — VAD blip
        now = utt["t1"]
        self._buf.recent.append(utt)
        evicted_n = 0
        while self._buf.recent and (now - self._buf.recent[0]["t1"]) > self.recent_window_s:
            self._buf.pending_for_summary.append(self._buf.recent.popleft())
            evicted_n += 1
        if len(self._buf.pending_for_summary) >= self.flush_every_n:
            self._buf.summary = await asyncio.to_thread(self._summarise)
        return self._snapshot(item_t=now)

    def _summarise(self) -> str:
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
            resp = self._client.generate(model=self.model, prompt=prompt, stream=False)
        except Exception as exc:  # noqa: BLE001
            # Network or model error — keep old summary, drop the block
            # so we don't infinitely retry the same poisoned batch.
            from os_helper import warning
            warning(f"GemmaAnalystStage: ollama call failed ({exc!r}); keeping previous summary")
            self._buf.pending_for_summary.clear()
            return self._buf.summary
        text = resp.get("response", "").strip() if isinstance(resp, dict) else str(resp).strip()
        self._buf.pending_for_summary.clear()
        return text

    def _snapshot(self, item_t: float | None) -> SummarySnapshot | None:
        if not self._buf.recent and not self._buf.summary:
            return None
        recent_block = "\n".join(
            f"[{u['t0']:.1f}-{u['t1']:.1f}] {u['speaker']}: {u['text']}"
            for u in self._buf.recent
        )
        t = item_t if item_t is not None else (
            self._buf.recent[-1]["t1"] if self._buf.recent else 0.0
        )
        return SummarySnapshot(
            t0=t,
            summary=self._buf.summary,
            recent=recent_block,
            model=self.model,
        )
