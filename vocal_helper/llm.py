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


def _extract_response_text(resp: object) -> str:
    """Pull the response text out of an Ollama ``generate`` reply.

    Different ``ollama`` package versions return either a plain dict
    or a Pydantic ``GenerateResponse``. Both have a ``"response"`` key
    or ``.response`` attribute.
    """
    if isinstance(resp, dict):
        return str(resp.get("response", "")).strip()
    text = getattr(resp, "response", None)
    if text is not None:
        return str(text).strip()
    # Last-resort fallback — the model dump.
    return str(resp).strip()


# Default selected by the 2026-06-30 7-model Pareto sweep
# (``studies/llm_model_size_sweep.py``, cadence ``flush_every_s=60``)
# on AMI IS1008a. Each model evaluated against its OWN single-shot
# offline-on-full-transcript summary :
#
#   model                RTF    cos_sim
#   gemma4:e2b-mlx     0.193    0.456
#   gemma4:e4b-mlx     0.313    0.420   ← initial default, now superseded
#   gemma4:12b-mlx     2.453    0.496   (Pareto on quality, too slow)
#   gemma3:4b          0.099    0.466   ← Pareto sweet spot, NEW DEFAULT
#   qwen2.5:3b         0.043    0.399   (Pareto on RTF, lower quality)
#   qwen3:8b           1.628    0.350   (dominated)
#   llama3.2:3b        0.066    0.367   (dominated)
#
# ``gemma3:4b`` dominates the prior default on BOTH axes simultaneously :
# 3 × faster (RTF 0.099 vs 0.313) AND higher cos_sim (0.466 vs 0.420).
# ``gemma4:12b-mlx`` is the absolute quality champion if the caller
# can afford RTF 2.45 (batch / offline use). ``qwen2.5:3b`` is the
# minimum-RTF pick at 0.043 — useful when running alongside a heavy
# TTS or on a constrained box.
DEFAULT_MODEL = "gemma3:4b"
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
        flush_every_s: float | None = DEFAULT_FLUSH_EVERY_S,
        host: str | None = None,
        prompt_template: str = DEFAULT_SUMMARY_PROMPT,
    ) -> None:
        self.model = model
        self.recent_window_s = recent_window_s
        self.flush_every_n = flush_every_n
        self.flush_every_s = flush_every_s
        self.host = host
        self.prompt_template = prompt_template
        self._client = None
        self._buf = _Buffer()
        # Track the t0 of the oldest pending-for-summary utterance so
        # the time-based cadence can fire on duration accumulated.
        self._oldest_pending_t0: float | None = None

    # ----- lifecycle ------------------------------------------------------

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import ollama  # type: ignore
        except ImportError as e:  # noqa: BLE001
            raise ImportError(
                "GemmaAnalystStage requires ollama. Install with `pip install vocal-helper[llm]`."
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
            span_s = newest_pending_t1 - (self._oldest_pending_t0 or newest_pending_t1)
            if span_s >= self.flush_every_s:
                should_flush = True
        elif len(self._buf.pending_for_summary) >= self.flush_every_n:
            should_flush = True
        if should_flush:
            self._buf.summary = await asyncio.to_thread(self._summarise)
            self._oldest_pending_t0 = None
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
        text = _extract_response_text(resp)
        self._buf.pending_for_summary.clear()
        return text

    def _snapshot(self, item_t: float | None) -> SummarySnapshot | None:
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
