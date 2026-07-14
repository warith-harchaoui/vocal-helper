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

# Module-scoped logger named ``vocal_helper.pipeline`` — callers (and tests)
# filter on this name to observe swallowed stage / subscriber exceptions, and
# ``exc_info=True`` attaches the real traceback to the record. os-helper's
# ``osh.warning`` logs to the root logger without a name or ``exc_info``, so it
# cannot express this named-logger-with-traceback contract ; stdlib logging is
# the right tool here.
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
        # Named logger + ``exc_info`` so the crash is observable with a full
        # traceback ; this silent-swallow path once deadlocked the suite.
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
            # Named logger + ``exc_info`` keeps the full traceback so a crashing
            # subscriber is diagnosable without breaking the fan-out.
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

    >>> import asyncio, vocal_helper as voh
    >>>
    >>> async def main():
    ...     pipeline = voh.Pipeline(
    ...         source=lambda: voh.sources.from_microphone(),
    ...         config=voh.PipelineConfig(
    ...             diar={"backend": "pyannote"},
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
        """Wire up queues, stages and subscriber lists from ``config``.

        Nothing runs here — construction only allocates the inter-stage queues
        and instantiates each stage. The chain starts on :meth:`run`.
        """
        self.source_factory = source
        self.config = config or PipelineConfig()

        # One bounded queue per hand-off. PCM gets the deep buffer (bursty,
        # 20 ms frames) ; the downstream segment queues stay shallow so a slow
        # ASR / LLM consumer back-pressures upstream instead of hoarding memory.
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
        tasks.append(asyncio.create_task(self._source_loop(), name="voh.source"))
        # VAD : q_pcm → q_voiced (with tee to subscribers).
        tasks.append(
            asyncio.create_task(
                self._vad.run(self._q_pcm, self._q_voiced),
                name="voh.vad",
            )
        )
        # Tee voiced subscribers + forward to diar (or to EOT then diar).
        q_voiced_for_diar: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        tasks.append(
            asyncio.create_task(
                self._tee(self._q_voiced, q_voiced_for_diar, self._voiced_subs),
                name="voh.tee.voiced",
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
                    name="voh.eot",
                )
            )
            tasks.append(
                asyncio.create_task(
                    self._diar.run(q_voiced_post_eot, self._q_diar),
                    name="voh.diar",
                )
            )
        else:
            tasks.append(
                asyncio.create_task(
                    self._diar.run(q_voiced_for_diar, self._q_diar),
                    name="voh.diar",
                )
            )
        # Tee diar subscribers + forward to ASR.
        q_diar_for_asr: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        tasks.append(
            asyncio.create_task(
                self._tee(self._q_diar, q_diar_for_asr, self._diar_subs),
                name="voh.tee.diar",
            )
        )
        tasks.append(
            asyncio.create_task(
                self._asr.run(q_diar_for_asr, self._q_utt),
                name="voh.asr",
            )
        )
        # Tee utterance subscribers + forward to LLM (if configured) and
        # to the yielded stream.
        q_utt_for_llm: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        q_utt_for_output: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        tasks.append(
            asyncio.create_task(
                self._tee_two(self._q_utt, q_utt_for_output, q_utt_for_llm, self._utt_subs),
                name="voh.tee.utt",
            )
        )
        if self._llm is not None:
            tasks.append(
                asyncio.create_task(
                    self._llm.run(q_utt_for_llm, self._q_summary),
                    name="voh.llm",
                )
            )
        else:
            # Drain to /dev/null so back-pressure doesn't stall the tee.
            tasks.append(
                asyncio.create_task(
                    self._drain(q_utt_for_llm),
                    name="voh.llm.disabled",
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
        """Pump frames from the source factory into ``q_pcm`` until it's exhausted."""
        try:
            async for frame in self.source_factory():
                await self._q_pcm.put(frame)
        finally:
            # Always cap the stream with the None sentinel — even if the source
            # raised — so the sentinel cascades and every stage shuts down.
            await self._q_pcm.put(None)

    async def _tee(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
        subscribers: list[Callable[..., Awaitable[None]]],
        *,
        stage: str = "tee",
    ) -> None:
        """Forward every item inbox→outbox, fanning each out to ``subscribers`` too."""
        while True:
            item = await inbox.get()
            # Forward first — the sentinel must propagate before we consider
            # returning, so the downstream stage always sees end-of-stream.
            await outbox.put(item)
            if item is None:
                return
            # Subscribers are observers only ; a crash in one is isolated + logged.
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
        """Fan every item to two outboxes (output + LLM) plus the subscriber list."""
        while True:
            item = await inbox.get()
            # Both branches must receive the sentinel, else the merger / LLM
            # stage would block forever waiting on a stream that never closes.
            await out_a.put(item)
            await out_b.put(item)
            if item is None:
                return
            await _invoke_subscribers(subscribers, item, stage)

    async def _drain(self, inbox: asyncio.Queue) -> None:
        """Swallow a queue to /dev/null so an unused branch can't back-pressure the tee."""
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

        # One-shot ``get`` wrapped as a task so ``asyncio.wait`` can race the
        # two queues and yield whichever event lands first.
        async def reader(q: asyncio.Queue):
            """Await and return the next item from ``q`` (one queue read)."""
            return await q.get()

        utt_task = asyncio.create_task(reader(utt_q))
        sum_task = asyncio.create_task(reader(summary_q))
        utt_done = False
        sum_done = False
        # Loop until BOTH streams have delivered their None sentinel — a closed
        # stream drops out of ``pending`` so we stop re-reading it.
        while not (utt_done and sum_done):
            pending = {t for t, done in [(utt_task, utt_done), (sum_task, sum_done)] if not done}
            if not pending:
                break
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                item = d.result()
                # For each finished read : a None retires that stream, otherwise
                # we yield the event and immediately re-arm a fresh read of it.
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

    >>> import asyncio, vocal_helper as voh
    >>>
    >>> async def main():
    ...     pipeline = voh.OfflinePipeline(
    ...         source=lambda: voh.sources.from_wav_file("meeting.wav",
    ...                                                  real_time=False),
    ...         config=voh.OfflinePipelineConfig(
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
        """Wire up the batch chain (no VAD): queues, offline diar, ASR, LLM."""
        self.source_factory = source
        self.config = config or OfflinePipelineConfig()

        # Same bounded-queue rationale as the streaming pipeline — deep PCM
        # buffer, shallow segment queues that back-pressure the diarizer.
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
        """Register an async callback fired for every :class:`DiarizedSegment`."""
        self._diar_subs.append(cb)

    def subscribe_utterances(self, cb: Callable[[Utterance], Awaitable[None]]) -> None:
        """Register an async callback fired for every :class:`Utterance`."""
        self._utt_subs.append(cb)

    async def run(self) -> AsyncIterator[Utterance | SummarySnapshot]:
        """Run the batch chain ; yield every Utterance and SummarySnapshot."""
        tasks: list[asyncio.Task] = []
        # Inbound — feed the whole PCM buffer to the offline diarizer.
        tasks.append(asyncio.create_task(self._source_loop(), name="voh.offline.source"))
        tasks.append(
            asyncio.create_task(
                self._diar.run(self._q_pcm, self._q_diar),
                name="voh.offline.diar",
            )
        )
        q_diar_for_asr: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        tasks.append(
            asyncio.create_task(
                self._tee(self._q_diar, q_diar_for_asr, self._diar_subs),
                name="voh.offline.tee.diar",
            )
        )
        tasks.append(
            asyncio.create_task(
                self._asr.run(q_diar_for_asr, self._q_utt),
                name="voh.offline.asr",
            )
        )
        q_utt_for_llm: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        q_utt_for_output: asyncio.Queue = asyncio.Queue(maxsize=self.config.qsize_seg)
        tasks.append(
            asyncio.create_task(
                self._tee_two(self._q_utt, q_utt_for_output, q_utt_for_llm, self._utt_subs),
                name="voh.offline.tee.utt",
            )
        )
        if self._llm is not None:
            tasks.append(
                asyncio.create_task(
                    self._llm.run(q_utt_for_llm, self._q_summary),
                    name="voh.offline.llm",
                )
            )
        else:
            # No analyst configured — drain the LLM branch and hand the merger
            # an immediate None so it never waits on a summary stream that
            # will never produce one.
            tasks.append(
                asyncio.create_task(
                    self._drain(q_utt_for_llm),
                    name="voh.offline.llm.disabled",
                )
            )
            await self._q_summary.put(None)

        try:
            async for ev in self._merge(q_utt_for_output, self._q_summary):
                yield ev
        finally:
            # Cancel any still-running stage, then await each so exceptions
            # (other than the expected CancelledError) get surfaced + logged.
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in tasks:
                await _await_task_swallow(t)

    # ----- internal coroutines (mirrors the streaming pipeline) ----------

    async def _source_loop(self) -> None:
        """Pump frames from the source factory into ``q_pcm`` until it's exhausted."""
        try:
            async for frame in self.source_factory():
                await self._q_pcm.put(frame)
        finally:
            # Sentinel-cap the stream even on source error, so diar shuts down.
            await self._q_pcm.put(None)

    async def _tee(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
        subscribers: list[Callable[..., Awaitable[None]]],
        *,
        stage: str = "tee",
    ) -> None:
        """Forward every item inbox→outbox, fanning each out to ``subscribers`` too."""
        while True:
            item = await inbox.get()
            # Propagate first so the sentinel always reaches the next stage.
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
        """Fan every item to two outboxes (output + LLM) plus the subscriber list."""
        while True:
            item = await inbox.get()
            # Both branches must see the sentinel or the merger blocks forever.
            await out_a.put(item)
            await out_b.put(item)
            if item is None:
                return
            await _invoke_subscribers(subscribers, item, stage)

    async def _drain(self, inbox: asyncio.Queue) -> None:
        """Swallow a queue to /dev/null so an unused branch can't back-pressure the tee."""
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

        # One-shot queue read as a task so ``asyncio.wait`` can race both queues.
        async def reader(q: asyncio.Queue):
            """Await and return the next item from ``q`` (one queue read)."""
            return await q.get()

        utt_task = asyncio.create_task(reader(utt_q))
        sum_task = asyncio.create_task(reader(summary_q))
        utt_done = False
        sum_done = False
        # Run until both streams have signalled end via their None sentinel.
        while not (utt_done and sum_done):
            pending = {t for t, done in [(utt_task, utt_done), (sum_task, sum_done)] if not done}
            if not pending:
                break
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                item = d.result()
                # None retires the stream ; a real event is yielded then the
                # read is re-armed to keep the interleave going.
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
