"""
vocal_helper.pipeline
=====================

The top-level producer/consumer orchestrator.

Stages, in order :

    [Source]   →  [VAD]  →  [Diar]  →  [ASR]   →  [LLM analyst]
       q0         q1       q2        q3          q4

Every arrow is an :class:`asyncio.Queue`. Each stage is a long-running
coroutine that ``await``s on its inbox, processes the event, and pushes
the result onto its outbox. Closing the upstream queue with a ``None``
sentinel cascades cleanly through the rest of the chain.

The pipeline is configured at construction time but only starts when
:meth:`Pipeline.start` is called. The caller can attach subscribers
to any intermediate queue (``q1`` for VAD events, ``q3`` for ASR
output, etc.) for live UI / WebSocket fan-out.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from vocal_helper.asr import WhisperStage
from vocal_helper.diar import OfflineDiarStage, OnlineDiarStage

# Optional in-flight module — the semantic EOT stage lives in Warith's
# WIP branch and is not always on disk. Guarded so the base pipeline
# stays importable when the module is absent.
try:
    from vocal_helper.eot import SemanticEOTStage  # type: ignore[assignment]
except Exception:  # pragma: no cover — optional
    SemanticEOTStage = None  # type: ignore[assignment]

from vocal_helper.llm import GemmaAnalystStage
from vocal_helper.types import (
    DiarizedSegment,
    PcmFrame,
    SummarySnapshot,
    Utterance,
    VoicedSegment,
)
from vocal_helper.vad import SileroVADStage

logger = logging.getLogger(__name__)


async def _await_task_swallow(t: asyncio.Task) -> None:
    """Await a cancelled / long-running task on pipeline shutdown.

    ``asyncio.CancelledError`` is expected — the task got the ``cancel()``
    we sent in the ``finally`` block. Anything else is a stage
    exception that has *not* been surfaced through the queue path (e.g.
    the diar backend raised on the last forward and we never emitted
    the ``None`` sentinel). Swallowing those silently is what let the
    pyannote 3.x ``DiarizeOutput`` API break wait undetected for hours ;
    always log them.
    """
    try:
        await t
    except asyncio.CancelledError:
        # Normal shutdown path.
        return
    except Exception:  # noqa: BLE001 — final barrier ; we log, then continue.
        logger.warning(
            "vocal_helper.pipeline: task %r crashed on shutdown",
            t.get_name(),
            exc_info=True,
        )


async def _invoke_subscribers(
    subscribers: list[Callable[..., Awaitable[None]]],
    item: object,
    stage: str,
) -> None:
    """Fan-out an item to user callbacks with per-callback isolation.

    A crashing subscriber must not break the pipeline — but the caller
    should still find out. We log a warning per failure with the stage
    name, callback name and full traceback ; the pipeline keeps
    forwarding to the remaining subscribers.
    """
    for cb in subscribers:
        try:
            await cb(item)
        except Exception:  # noqa: BLE001 — user code ; log, then continue.
            logger.warning(
                "vocal_helper.pipeline: subscriber %r on stage %r raised",
                getattr(cb, "__qualname__", repr(cb)),
                stage,
                exc_info=True,
            )


# Bounded queue sizes — large enough to absorb a ~ 1 s burst, small
# enough that back-pressure pushes upstream before memory explodes.
_DEFAULT_QSIZE_PCM = 200  # 200 × 20 ms = 4 s of audio in flight
_DEFAULT_QSIZE_SEG = 32  # plenty for a slow ASR / LLM consumer


SourceFactory = Callable[[], AsyncIterator[PcmFrame]]


@dataclass
class PipelineConfig:
    """Configuration object for :class:`Pipeline`.

    Pass per-stage settings as dicts. The pipeline forwards them
    verbatim to each stage's constructor.
    """

    vad: dict = field(default_factory=dict)
    # ``None`` disables the semantic end-of-turn gating stage —
    # default behaviour matches the v0.1.0 release where Silero VAD's
    # silence threshold is the only turn-end signal. Provide a dict to
    # enable :class:`SemanticEOTStage` between VAD and diarization
    # (cuts ~ 39 % of mid-sentence breaks per the LiveKit turn-detector
    # white paper, at the cost of one extra LLM hop per voiced segment).
    eot: dict | None = None
    diar: dict = field(default_factory=dict)
    asr: dict = field(default_factory=dict)
    # ``None`` disables the LLM analyst — useful when you only need
    # the transcript without summarisation.
    llm: dict | None = None
    qsize_pcm: int = _DEFAULT_QSIZE_PCM
    qsize_seg: int = _DEFAULT_QSIZE_SEG


class Pipeline:
    """End-to-end audio→text(→summary) producer/consumer chain.

    Usage
    -----

    >>> import asyncio, vocal_helper as vh
    >>>
    >>> async def main():
    ...     pipeline = vh.Pipeline(
    ...         source=lambda: vh.sources.from_microphone(),
    ...         config=vh.PipelineConfig(
    ...             diar={"backend": "pyannote", "hf_token": "hf_..."},
    ...             asr={"model": "large-v3-turbo-q5_0"},
    ...             llm={"model": "gemma4:e4b"},
    ...         ),
    ...     )
    ...     async for event in pipeline.run():
    ...         print(event)
    ...
    >>> asyncio.run(main())

    Yielded events are a mix of :class:`Utterance` and
    :class:`SummarySnapshot` (the latter only when ``llm`` is
    configured). For per-stage observation see
    :meth:`Pipeline.subscribe_voiced` / ``subscribe_diarized``.
    """

    def __init__(
        self,
        *,
        source: SourceFactory,
        config: PipelineConfig | None = None,
    ) -> None:
        self.source_factory = source
        self.config = config or PipelineConfig()

        self._q_pcm: asyncio.Queue[PcmFrame | None] = asyncio.Queue(maxsize=self.config.qsize_pcm)
        self._q_voiced: asyncio.Queue[VoicedSegment | None] = asyncio.Queue(
            maxsize=self.config.qsize_seg
        )
        self._q_diar: asyncio.Queue[DiarizedSegment | None] = asyncio.Queue(
            maxsize=self.config.qsize_seg
        )
        self._q_utt: asyncio.Queue[Utterance | None] = asyncio.Queue(maxsize=self.config.qsize_seg)
        self._q_summary: asyncio.Queue[SummarySnapshot | None] = asyncio.Queue(
            maxsize=self.config.qsize_seg
        )

        self._vad = SileroVADStage(**self.config.vad)
        self._eot: SemanticEOTStage | None = (
            SemanticEOTStage(**self.config.eot) if self.config.eot is not None else None
        )
        self._diar = OnlineDiarStage(**self.config.diar)
        # Streaming enables whisper warm-up by default so the first
        # caption doesn't stall on the model's cold first inference. Any
        # explicit ``warmup`` in ``config.asr`` wins ; the offline
        # pipeline leaves it off (batch throughput doesn't care).
        self._asr = WhisperStage(**{"warmup": True, **self.config.asr})
        self._llm: GemmaAnalystStage | None = (
            GemmaAnalystStage(**self.config.llm) if self.config.llm is not None else None
        )

        self._voiced_subs: list[Callable[[VoicedSegment], Awaitable[None]]] = []
        self._diar_subs: list[Callable[[DiarizedSegment], Awaitable[None]]] = []
        self._utt_subs: list[Callable[[Utterance], Awaitable[None]]] = []

    # ----- public API ----------------------------------------------------

    def subscribe_voiced(self, cb: Callable[[VoicedSegment], Awaitable[None]]) -> None:
        """Async callback for every :class:`VoicedSegment` after VAD."""
        self._voiced_subs.append(cb)

    def subscribe_diarized(self, cb: Callable[[DiarizedSegment], Awaitable[None]]) -> None:
        """Async callback for every :class:`DiarizedSegment` after diarization."""
        self._diar_subs.append(cb)

    def subscribe_utterances(self, cb: Callable[[Utterance], Awaitable[None]]) -> None:
        """Async callback for every :class:`Utterance` after ASR."""
        self._utt_subs.append(cb)

    async def run(self) -> AsyncIterator[Utterance | SummarySnapshot]:
        """Run the pipeline ; yield every Utterance and SummarySnapshot."""
        tasks: list[asyncio.Task] = []
        # Inbound — read the source and push frames into q_pcm.
        tasks.append(asyncio.create_task(self._source_loop(), name="vh.source"))
        # VAD : q_pcm → q_voiced (with tee to subscribers).
        tasks.append(
            asyncio.create_task(
                self._vad.run(self._q_pcm, self._q_voiced),
                name="vh.vad",
            )
        )
        # Tee voiced subscribers + forward to diar (or to EOT then diar).
        q_voiced_for_diar: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        tasks.append(
            asyncio.create_task(
                self._tee(self._q_voiced, q_voiced_for_diar, self._voiced_subs),
                name="vh.tee.voiced",
            )
        )
        if self._eot is not None:
            # Insert semantic EOT gating between VAD and diarization —
            # holds back segments that look mid-thought, merges them
            # with their successor, and forwards the merged
            # super-segment downstream. The diar queue therefore
            # sees fewer, larger segments at higher semantic completeness.
            q_voiced_post_eot: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
            tasks.append(
                asyncio.create_task(
                    self._eot.run(q_voiced_for_diar, q_voiced_post_eot),
                    name="vh.eot",
                )
            )
            tasks.append(
                asyncio.create_task(
                    self._diar.run(q_voiced_post_eot, self._q_diar),
                    name="vh.diar",
                )
            )
        else:
            tasks.append(
                asyncio.create_task(
                    self._diar.run(q_voiced_for_diar, self._q_diar),
                    name="vh.diar",
                )
            )
        # Tee diar subscribers + forward to ASR.
        q_diar_for_asr: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        tasks.append(
            asyncio.create_task(
                self._tee(self._q_diar, q_diar_for_asr, self._diar_subs),
                name="vh.tee.diar",
            )
        )
        tasks.append(
            asyncio.create_task(
                self._asr.run(q_diar_for_asr, self._q_utt),
                name="vh.asr",
            )
        )
        # Tee utterance subscribers + forward to LLM (if configured) and
        # to the yielded stream.
        q_utt_for_llm: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        q_utt_for_output: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        tasks.append(
            asyncio.create_task(
                self._tee_two(self._q_utt, q_utt_for_output, q_utt_for_llm, self._utt_subs),
                name="vh.tee.utt",
            )
        )
        if self._llm is not None:
            tasks.append(
                asyncio.create_task(
                    self._llm.run(q_utt_for_llm, self._q_summary),
                    name="vh.llm",
                )
            )
        else:
            # Drain to /dev/null so back-pressure doesn't stall the tee.
            tasks.append(
                asyncio.create_task(
                    self._drain(q_utt_for_llm),
                    name="vh.llm.disabled",
                )
            )
            # Push an immediate None into q_summary so the merger knows
            # there is no summary stream.
            await self._q_summary.put(None)

        try:
            async for ev in self._merge(q_utt_for_output, self._q_summary):
                yield ev
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in tasks:
                await _await_task_swallow(t)

    # ----- internal coroutines ------------------------------------------

    async def _source_loop(self) -> None:
        try:
            async for frame in self.source_factory():
                await self._q_pcm.put(frame)
        finally:
            await self._q_pcm.put(None)

    async def _tee(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
        subscribers: list[Callable[..., Awaitable[None]]],
        *,
        stage: str = "tee",
    ) -> None:
        while True:
            item = await inbox.get()
            await outbox.put(item)
            if item is None:
                return
            await _invoke_subscribers(subscribers, item, stage)

    async def _tee_two(
        self,
        inbox: asyncio.Queue,
        out_a: asyncio.Queue,
        out_b: asyncio.Queue,
        subscribers: list[Callable[..., Awaitable[None]]],
        *,
        stage: str = "tee_two",
    ) -> None:
        while True:
            item = await inbox.get()
            await out_a.put(item)
            await out_b.put(item)
            if item is None:
                return
            await _invoke_subscribers(subscribers, item, stage)

    async def _drain(self, inbox: asyncio.Queue) -> None:
        while True:
            item = await inbox.get()
            if item is None:
                return

    async def _merge(
        self,
        utt_q: asyncio.Queue,
        summary_q: asyncio.Queue,
    ) -> AsyncIterator[Utterance | SummarySnapshot]:
        """Interleave Utterance + SummarySnapshot streams until both end."""

        async def reader(q: asyncio.Queue):
            return await q.get()

        utt_task = asyncio.create_task(reader(utt_q))
        sum_task = asyncio.create_task(reader(summary_q))
        utt_done = False
        sum_done = False
        while not (utt_done and sum_done):
            pending = {t for t, done in [(utt_task, utt_done), (sum_task, sum_done)] if not done}
            if not pending:
                break
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                item = d.result()
                if d is utt_task:
                    if item is None:
                        utt_done = True
                    else:
                        yield item
                        utt_task = asyncio.create_task(reader(utt_q))
                elif d is sum_task:
                    if item is None:
                        sum_done = True
                    else:
                        yield item
                        sum_task = asyncio.create_task(reader(summary_q))


# ===========================================================================
# OFFLINE PIPELINE
# ===========================================================================


@dataclass
class OfflinePipelineConfig:
    """Configuration object for :class:`OfflinePipeline`.

    Same shape as :class:`PipelineConfig` minus the streaming-specific
    ``vad`` block — the offline diarizer ingests the full audio and
    relies on the backend's own VAD / segmentation.
    """

    diar: dict = field(default_factory=dict)
    asr: dict = field(default_factory=dict)
    llm: dict | None = None
    qsize_pcm: int = _DEFAULT_QSIZE_PCM
    qsize_seg: int = _DEFAULT_QSIZE_SEG


class OfflinePipeline:
    """End-to-end batch chain : source → offline diar → ASR → LLM.

    Trades the live VAD + per-segment online clustering for a single
    call to :class:`OfflineDiarStage` on the full PCM buffer (with
    long-form chunking baked in). This is the **best-quality** path
    when the input is fully available — typical use cases : meeting
    recordings, podcasts, voicemail batches, lecture archives.

    Usage
    -----

    >>> import asyncio, vocal_helper as vh
    >>>
    >>> async def main():
    ...     pipeline = vh.OfflinePipeline(
    ...         source=lambda: vh.sources.from_wav_file("meeting.wav",
    ...                                                  real_time=False),
    ...         config=vh.OfflinePipelineConfig(
    ...             diar={"backend": "pyannote"},
    ...             asr={"language": "en"},
    ...             llm={"model": "gemma4:e4b"},
    ...         ),
    ...     )
    ...     async for ev in pipeline.run():
    ...         print(ev)
    ...
    >>> asyncio.run(main())

    The cadence trade
    -----------------
    Because the four stages are decoupled by queues, each can run at
    its own pace : the diarizer waits for the entire PCM, the ASR
    streams through diarized segments as the diarizer finishes
    emitting them, the LLM analyst aggregates utterances as they land.
    The only end-to-end blocker is the offline diarizer itself.
    """

    def __init__(
        self,
        *,
        source: SourceFactory,
        config: OfflinePipelineConfig | None = None,
    ) -> None:
        self.source_factory = source
        self.config = config or OfflinePipelineConfig()

        self._q_pcm: asyncio.Queue[PcmFrame | None] = asyncio.Queue(maxsize=self.config.qsize_pcm)
        self._q_diar: asyncio.Queue[DiarizedSegment | None] = asyncio.Queue(
            maxsize=self.config.qsize_seg
        )
        self._q_utt: asyncio.Queue[Utterance | None] = asyncio.Queue(maxsize=self.config.qsize_seg)
        self._q_summary: asyncio.Queue[SummarySnapshot | None] = asyncio.Queue(
            maxsize=self.config.qsize_seg
        )

        self._diar = OfflineDiarStage(**self.config.diar)
        # Offline defaults to the full-throttle batched ASR path (one
        # whisper call per concatenated ``max_chunk_s`` window instead of
        # one per diarized segment ; ~6× lower RTF at on-par WER per the
        # 2026-07-09 sweep). Callers opt back into the per-segment path
        # with ``OfflinePipelineConfig(asr={"batch": False})`` ; any
        # explicit ``batch`` in ``config.asr`` wins.
        asr_cfg = {"batch": True, **self.config.asr}
        self._asr = WhisperStage(**asr_cfg)
        self._llm: GemmaAnalystStage | None = (
            GemmaAnalystStage(**self.config.llm) if self.config.llm is not None else None
        )

        self._diar_subs: list[Callable[[DiarizedSegment], Awaitable[None]]] = []
        self._utt_subs: list[Callable[[Utterance], Awaitable[None]]] = []

    # ----- public API ----------------------------------------------------

    def subscribe_diarized(self, cb: Callable[[DiarizedSegment], Awaitable[None]]) -> None:
        self._diar_subs.append(cb)

    def subscribe_utterances(self, cb: Callable[[Utterance], Awaitable[None]]) -> None:
        self._utt_subs.append(cb)

    async def run(self) -> AsyncIterator[Utterance | SummarySnapshot]:
        tasks: list[asyncio.Task] = []
        tasks.append(asyncio.create_task(self._source_loop(), name="vh.offline.source"))
        tasks.append(
            asyncio.create_task(
                self._diar.run(self._q_pcm, self._q_diar),
                name="vh.offline.diar",
            )
        )
        q_diar_for_asr: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        tasks.append(
            asyncio.create_task(
                self._tee(self._q_diar, q_diar_for_asr, self._diar_subs),
                name="vh.offline.tee.diar",
            )
        )
        tasks.append(
            asyncio.create_task(
                self._asr.run(q_diar_for_asr, self._q_utt),
                name="vh.offline.asr",
            )
        )
        q_utt_for_llm: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        q_utt_for_output: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        tasks.append(
            asyncio.create_task(
                self._tee_two(self._q_utt, q_utt_for_output, q_utt_for_llm, self._utt_subs),
                name="vh.offline.tee.utt",
            )
        )
        if self._llm is not None:
            tasks.append(
                asyncio.create_task(
                    self._llm.run(q_utt_for_llm, self._q_summary),
                    name="vh.offline.llm",
                )
            )
        else:
            tasks.append(
                asyncio.create_task(
                    self._drain(q_utt_for_llm),
                    name="vh.offline.llm.disabled",
                )
            )
            await self._q_summary.put(None)

        try:
            async for ev in self._merge(q_utt_for_output, self._q_summary):
                yield ev
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in tasks:
                await _await_task_swallow(t)

    # ----- internal coroutines (mirrors the streaming pipeline) ----------

    async def _source_loop(self) -> None:
        try:
            async for frame in self.source_factory():
                await self._q_pcm.put(frame)
        finally:
            await self._q_pcm.put(None)

    async def _tee(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
        subscribers: list[Callable[..., Awaitable[None]]],
        *,
        stage: str = "tee",
    ) -> None:
        while True:
            item = await inbox.get()
            await outbox.put(item)
            if item is None:
                return
            await _invoke_subscribers(subscribers, item, stage)

    async def _tee_two(
        self,
        inbox: asyncio.Queue,
        out_a: asyncio.Queue,
        out_b: asyncio.Queue,
        subscribers: list[Callable[..., Awaitable[None]]],
        *,
        stage: str = "tee_two",
    ) -> None:
        while True:
            item = await inbox.get()
            await out_a.put(item)
            await out_b.put(item)
            if item is None:
                return
            await _invoke_subscribers(subscribers, item, stage)

    async def _drain(self, inbox: asyncio.Queue) -> None:
        while True:
            item = await inbox.get()
            if item is None:
                return

    async def _merge(
        self,
        utt_q: asyncio.Queue,
        summary_q: asyncio.Queue,
    ) -> AsyncIterator[Utterance | SummarySnapshot]:
        async def reader(q: asyncio.Queue):
            return await q.get()

        utt_task = asyncio.create_task(reader(utt_q))
        sum_task = asyncio.create_task(reader(summary_q))
        utt_done = False
        sum_done = False
        while not (utt_done and sum_done):
            pending = {t for t, done in [(utt_task, utt_done), (sum_task, sum_done)] if not done}
            if not pending:
                break
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                item = d.result()
                if d is utt_task:
                    if item is None:
                        utt_done = True
                    else:
                        yield item
                        utt_task = asyncio.create_task(reader(utt_q))
                elif d is sum_task:
                    if item is None:
                        sum_done = True
                    else:
                        yield item
                        sum_task = asyncio.create_task(reader(summary_q))
