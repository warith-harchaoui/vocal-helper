"""
vocal_helper.asr
================

pywhispercpp-turbo ASR stage. Consumes :class:`DiarizedSegment`
and emits :class:`Utterance` once whisper.cpp returns.

Concurrency model
-----------------
whisper.cpp is **CPU-bound and blocking** — running it inline in
the event loop would stall every other stage. The stage delegates
each transcription to :func:`asyncio.to_thread` so the loop stays
free for VAD / diarization / LLM analyst work.

For very short utterances (< ``min_segment_ms``, default 250 ms)
we skip whisper entirely and emit an empty-text :class:`Utterance`.
Whisper's hallucination rate at sub-300 ms inputs is unacceptable
for live captioning.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vocal_helper.types import DiarizedSegment, Utterance

DEFAULT_MODEL = "large-v3-turbo-q5_0"
DEFAULT_LANGUAGE = "auto"
DEFAULT_THREADS = 6
DEFAULT_MIN_SEGMENT_MS = 250
# Offline "full-throttle" batching (opt-in via ``batch=True``).
# whisper.cpp pads every mel to a 30 s window, so a 0.8 s turn costs
# nearly the same encoder pass as a 25 s one. Concatenating consecutive
# diarized segments into one ≤ ``max_chunk_s`` call amortises that fixed
# cost — the 2026-07-09 sweep (``studies/asr_offline_batching.py``) on
# 12 bagarre-rich bilingual mixes measured cross-speaker packing at 24 s
# **6.5× lower RTF (0.054 vs 0.353) at *better* WER (0.565 vs 0.612)** :
# the longer decoder context cuts short-segment hallucination faster
# than the occasional intra-chunk language switch costs. Speaker-coherent
# packing was a wash (speakers alternate, so nothing groups). Attribution
# is preserved by re-mapping each returned phrase back to the diarized
# segment whose local time window contains it — so the stage still emits
# exactly one :class:`Utterance` per input :class:`DiarizedSegment`.
DEFAULT_MAX_CHUNK_S = 24.0
_BATCH_PAD_S = 0.1  # silence inserted between concatenated segments
# Bias prompt — conditioning text whisper sees BEFORE each utterance.
# Empirical impact on AMI dev-slice (2026-06-30 study
# ``studies/whisper_prompt_lang_lock.py``) :
#   - WER drops 15-25 percentage points on a 16-min meeting when a
#     domain-aligned bias prompt is supplied ;
#   - RTF improves up to 39 % because whisper spends less time on
#     hallucinated digressions.
# Default is empty (generic transcription) ; callers SHOULD provide a
# short prompt naming the domain vocabulary expected in the audio.
DEFAULT_INITIAL_PROMPT = ""


class WhisperStage:
    """Producer/consumer pywhispercpp transcription stage.

    Parameters
    ----------
    model : str
        whisper.cpp model name. Default ``"large-v3-turbo-q5_0"``
        — the cheapest word-timestamp-capable variant.
    language : str
        ISO-639-1 code (``"en"``, ``"fr"``, …) or ``"auto"`` for
        language identification.
    threads : int
        CPU threads handed to whisper.cpp.
    word_timestamps : bool
        Emit per-word timestamps. Default ``True`` — matches the
        ``Utterance.words`` contract.
    initial_prompt : str
        Bias / vocabulary prompt passed to whisper before every
        transcription. Empty by default. **Strongly recommended** :
        the 2026-06-30 sweep on AMI dev-slice
        (``studies/whisper_prompt_lang_lock.py``) showed a domain-
        aligned prompt cuts WER by 15-25 percentage points and saves
        up to 39 % RTF. Good prompts name the conversational domain
        and a handful of expected proper nouns / technical terms,
        e.g. ``"AMI meeting transcript: remote control design, "
        "marketing plan, user interface, requirements."``
    min_segment_ms : int
        Drop segments shorter than this — whisper hallucinates on
        very short inputs.
    batch : bool
        **Offline full-throttle mode.** When ``True`` the stage stops
        transcribing one diarized segment at a time and instead packs
        consecutive segments into ``max_chunk_s`` windows, running a
        single whisper call per window (whisper.cpp pads every call to a
        fixed 30 s mel, so fewer/fuller calls amortise that cost). Each
        returned phrase is re-mapped back to the diarized segment whose
        local time window contains it, so the output contract is
        unchanged — still one :class:`Utterance` per input segment, with
        the diarizer's speaker id preserved. Off by default ; the
        streaming :class:`~vocal_helper.pipeline.Pipeline` never sets it.
        The 2026-07-09 sweep (``studies/asr_offline_batching.py``)
        measured 6.5× lower RTF *and* better WER vs the per-segment path.
    max_chunk_s : float
        Max concatenated-audio length per batched whisper call. Only
        used when ``batch=True``. Default 24 s (fills the 30 s window
        well — the sweep's Pareto winner). Lowering toward 12 s trades a
        little speed for safety if inputs switch language every few
        seconds (a chunk straddling an en→fr switch forces one language).
    warmup : bool
        When ``True``, :meth:`run` runs one throwaway inference on
        silence before consuming the queue, moving whisper's ~1 s
        first-inference stall off the streaming hot path. Off by default ;
        the streaming :class:`~vocal_helper.pipeline.Pipeline` enables it.

    Notes
    -----
    The pywhispercpp model is loaded lazily on the first frame so
    cold pipelines don't pay the ~ 1.5 GB download until they
    actually need it.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        threads: int = DEFAULT_THREADS,
        word_timestamps: bool = True,
        initial_prompt: str = DEFAULT_INITIAL_PROMPT,
        min_segment_ms: int = DEFAULT_MIN_SEGMENT_MS,
        batch: bool = False,
        max_chunk_s: float = DEFAULT_MAX_CHUNK_S,
        warmup: bool = False,
    ) -> None:
        """Configure the whisper stage ; the model itself loads lazily.

        Parameters
        ----------
        model : str
            pywhispercpp model name / path. Default :data:`DEFAULT_MODEL`.
        language : str
            Transcription language, or ``"auto"`` to let whisper detect
            it. Default :data:`DEFAULT_LANGUAGE`.
        threads : int
            Number of CPU threads for whisper.cpp. Default
            :data:`DEFAULT_THREADS`.
        word_timestamps : bool
            Emit per-word timestamps to fill ``Utterance.words``. Default
            ``True``.
        initial_prompt : str
            Bias / vocabulary prompt passed before each transcription.
            Default :data:`DEFAULT_INITIAL_PROMPT`.
        min_segment_ms : int
            Drop segments shorter than this — whisper hallucinates on very
            short inputs. Default :data:`DEFAULT_MIN_SEGMENT_MS`.
        batch : bool
            Enable the offline full-throttle path (pack consecutive
            segments into ``max_chunk_s`` windows). Off by default so the
            streaming path and every existing caller keep the per-segment
            behaviour.
        max_chunk_s : float
            Max concatenated-audio length per batched whisper call ; only
            used when ``batch=True``. Default :data:`DEFAULT_MAX_CHUNK_S`.
        warmup : bool
            When ``True``, :meth:`run` front-loads whisper's first (slow)
            inference on silence before consuming the queue. Off by
            default ; the streaming pipeline turns it on.
        """
        self.model_name = model
        self.language = language
        self.threads = threads
        self.word_timestamps = word_timestamps
        self.initial_prompt = initial_prompt
        self.min_segment_ms = min_segment_ms
        # ``batch`` turns on the offline full-throttle path (see
        # ``DEFAULT_MAX_CHUNK_S``). Off by default so the streaming
        # pipeline and every existing caller keep the exact per-segment
        # behaviour — one whisper call, one Utterance, per segment.
        self.batch = batch
        self.max_chunk_s = max_chunk_s
        # ``warmup`` front-loads whisper's first (slow) inference during
        # pipeline start-up instead of paying it on the first real
        # segment — a pure latency win for the streaming path, where the
        # first caption otherwise stalls ~1 s while the graph warms.
        # Off by default (offline batch doesn't care) ; the streaming
        # :class:`~vocal_helper.pipeline.Pipeline` turns it on.
        self.warmup = warmup
        self._model: Any | None = None

    # ----- lifecycle ------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily instantiate the pywhispercpp model on first use.

        Idempotent — returns immediately if the model is already loaded,
        so it is safe to call at the top of every entry point.

        Raises
        ------
        ImportError
            If the ``pywhispercpp`` backend is not installed.
        """
        if self._model is not None:
            return
        try:
            from pywhispercpp.model import Model  # type: ignore
        except ImportError as e:  # noqa: BLE001
            raise ImportError(
                "WhisperStage requires pywhispercpp. "
                "Install with `pip install pywhispercpp` or "
                "`pip install vocal-helper`."
            ) from e
        kwargs: dict[str, Any] = {
            "n_threads": self.threads,
            "print_realtime": False,
            "print_progress": False,
            "token_timestamps": self.word_timestamps,
        }
        if self.language != "auto":
            kwargs["language"] = self.language
        self._model = Model(self.model_name, **kwargs)

    def _warm(self) -> None:
        """Run one throwaway inference on 0.5 s of silence (blocking).

        Best-effort — a warm-up failure must never stop the pipeline, so
        we swallow any backend error and let the first real segment
        surface it through the normal path.
        """
        self._ensure_model()
        with contextlib.suppress(Exception):  # warm-up is optional.
            self._model.transcribe(np.zeros(8000, dtype=np.float32))

    # ----- public coroutine ----------------------------------------------

    async def run(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
    ) -> None:
        """Consume :class:`DiarizedSegment` from ``inbox``, push :class:`Utterance`."""
        self._ensure_model()
        if self.warmup:
            await asyncio.to_thread(self._warm)
        if self.batch:
            await self._run_batched(inbox, outbox)
            return
        while True:
            item = await inbox.get()
            if item is None:
                await outbox.put(None)
                return
            utt = await asyncio.to_thread(self._transcribe_blocking, item)
            if utt is not None:
                await outbox.put(utt)

    async def _run_batched(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
    ) -> None:
        """Full-throttle path — pack consecutive segments, one call per chunk.

        Segments are drained from ``inbox`` continuously (so the upstream
        diarizer never blocks on a full queue), buffered, then packed
        into ``max_chunk_s`` windows. Each chunk is transcribed in one
        whisper call and split back into one :class:`Utterance` per input
        segment, preserving order, timing and the diarizer's speaker id.
        """
        buf: list[DiarizedSegment] = []
        while True:
            item = await inbox.get()
            if item is None:
                break
            buf.append(item)
        for chunk in _pack_segments(buf, self.max_chunk_s):
            utts = await asyncio.to_thread(self._transcribe_chunk_blocking, chunk)
            for utt in utts:
                await outbox.put(utt)
        await outbox.put(None)

    # ----- the blocking part --------------------------------------------

    def _transcribe_blocking(self, seg: DiarizedSegment) -> Utterance | None:
        """Whisper call. Runs in a worker thread."""
        dur_ms = (seg["t1"] - seg["t0"]) * 1000.0
        if dur_ms < self.min_segment_ms:
            return Utterance(
                t0=seg["t0"],
                t1=seg["t1"],
                speaker=seg["speaker"],
                text="",
                words=[],
                language=None,
            )
        try:
            if self.initial_prompt:
                segments = self._model.transcribe(
                    seg["pcm"],
                    initial_prompt=self.initial_prompt,
                )
            else:
                segments = self._model.transcribe(seg["pcm"])
        except Exception:  # noqa: BLE001
            return None
        text_parts: list[str] = []
        words: list[tuple[float, float, str]] = []
        language: str | None = None
        for s in segments:
            text_parts.append(s.text.strip())
            # ``Segment.t0`` / ``t1`` come back in centiseconds — convert.
            seg_t0 = seg["t0"] + float(s.t0) / 100.0
            seg_t1 = seg["t0"] + float(s.t1) / 100.0
            words.append((seg_t0, seg_t1, s.text.strip()))
            if language is None and getattr(s, "language", None):
                language = s.language
        text = " ".join(p for p in text_parts if p).strip()
        return Utterance(
            t0=seg["t0"],
            t1=seg["t1"],
            speaker=seg["speaker"],
            text=text,
            words=words,
            language=language,
        )

    # ----- batched blocking part ----------------------------------------

    @staticmethod
    def _empty_utt(seg: DiarizedSegment) -> Utterance:
        """Build an empty :class:`Utterance` carrying a segment's metadata.

        Used when a segment produces no transcription (too short, or no
        phrase mapped to it) but must still be emitted so the output
        contract stays one :class:`Utterance` per input segment.

        Parameters
        ----------
        seg : DiarizedSegment
            The source segment ; its ``t0`` / ``t1`` / ``speaker`` are
            copied over, with empty ``text`` / ``words`` and no language.

        Returns
        -------
        Utterance
            An utterance with the segment's timing and speaker but no text.
        """
        return Utterance(
            t0=seg["t0"],
            t1=seg["t1"],
            speaker=seg["speaker"],
            text="",
            words=[],
            language=None,
        )

    def _transcribe_chunk_blocking(self, chunk: list[DiarizedSegment]) -> list[Utterance]:
        """Transcribe a concatenated chunk, split back to per-segment Utterances.

        Lays every segment's PCM end-to-end (``_BATCH_PAD_S`` of silence
        between them), runs one whisper call, then assigns each returned
        phrase to the segment whose local time window contains its
        midpoint. Returns exactly one :class:`Utterance` per input
        segment, in input order — so the batched path is contract-
        identical to the per-segment path from the consumer's side.
        """
        sr = chunk[0]["sample_rate"]
        pad = np.zeros(int(_BATCH_PAD_S * sr), dtype=np.float32)
        pieces: list[NDArray[np.float32]] = []
        # windows[i] = (local_t0, local_t1) of chunk[i] in the concatenated buffer.
        windows: list[tuple[float, float]] = []
        cursor = 0.0
        for i, seg in enumerate(chunk):
            if i:
                pieces.append(pad)
                cursor += pad.shape[0] / sr
            dur = seg["pcm"].shape[0] / sr
            windows.append((cursor, cursor + dur))
            pieces.append(seg["pcm"])
            cursor += dur
        pcm = np.concatenate(pieces).astype(np.float32, copy=False)

        try:
            if self.initial_prompt:
                segments = self._model.transcribe(pcm, initial_prompt=self.initial_prompt)
            else:
                segments = self._model.transcribe(pcm)
        except Exception:  # noqa: BLE001 — keep the contract : one Utterance per seg.
            return [self._empty_utt(seg) for seg in chunk]

        acc: list[dict[str, Any]] = [{"text": [], "words": [], "language": None} for _ in chunk]
        for s in segments:
            # ``Segment.t0`` / ``t1`` are chunk-local centiseconds.
            st0 = float(s.t0) / 100.0
            st1 = float(s.t1) / 100.0
            idx = _assign_window(windows, (st0 + st1) / 2.0)
            wl0 = windows[idx][0]
            seg = chunk[idx]
            text = s.text.strip()
            abs_t0 = seg["t0"] + max(0.0, st0 - wl0)
            abs_t1 = seg["t0"] + max(0.0, st1 - wl0)
            acc[idx]["text"].append(text)
            acc[idx]["words"].append((abs_t0, abs_t1, text))
            if acc[idx]["language"] is None and getattr(s, "language", None):
                acc[idx]["language"] = s.language

        utts: list[Utterance] = []
        for seg, a in zip(chunk, acc, strict=True):
            utts.append(
                Utterance(
                    t0=seg["t0"],
                    t1=seg["t1"],
                    speaker=seg["speaker"],
                    text=" ".join(p for p in a["text"] if p).strip(),
                    words=a["words"],
                    language=a["language"],
                )
            )
        return utts


# ---------------------------------------------------------------------------
# Batching helpers (module-level — pure, unit-testable without whisper).
# ---------------------------------------------------------------------------


def _pack_segments(segs: list[DiarizedSegment], max_chunk_s: float) -> list[list[DiarizedSegment]]:
    """Greedily group consecutive segments into ≤ ``max_chunk_s`` chunks.

    Cross-speaker on purpose : the 2026-07-09 sweep showed speaker-
    coherent packing barely groups anything when speakers alternate
    (so it yields no speedup), while cross-speaker packing amortises the
    fixed whisper cost — speaker identity is recovered downstream by
    time-window re-mapping, not by the chunk boundary.
    """
    chunks: list[list[DiarizedSegment]] = []
    cur: list[DiarizedSegment] = []
    cur_dur = 0.0
    for seg in segs:
        d = seg["pcm"].shape[0] / float(seg["sample_rate"])
        if cur and cur_dur + _BATCH_PAD_S + d > max_chunk_s:
            chunks.append(cur)
            cur, cur_dur = [], 0.0
        cur.append(seg)
        cur_dur += (_BATCH_PAD_S if cur_dur else 0.0) + d
    if cur:
        chunks.append(cur)
    return chunks


def _assign_window(windows: list[tuple[float, float]], t: float) -> int:
    """Index of the window containing ``t`` ; nearest by edge distance if none."""
    for i, (lo, hi) in enumerate(windows):
        if lo <= t <= hi:
            return i
    return min(
        range(len(windows)),
        key=lambda i: min(abs(t - windows[i][0]), abs(t - windows[i][1])),
    )


# ---------------------------------------------------------------------------
# Helper — synchronous transcribe of a single PCM buffer, for tests.
# ---------------------------------------------------------------------------


def transcribe_pcm(
    pcm: NDArray[np.float32],
    sr: int,
    *,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    threads: int = DEFAULT_THREADS,
    initial_prompt: str = DEFAULT_INITIAL_PROMPT,
) -> str:
    """Synchronous one-shot transcription. Loads whisper.cpp on call.

    ``initial_prompt`` is the same domain-bias lever as
    :class:`WhisperStage` — empty by default, but strongly recommended
    (cuts WER 15-25 pp on AMI, saves up to 39 % RTF).
    """
    stage = WhisperStage(
        model=model,
        language=language,
        threads=threads,
        word_timestamps=False,
        initial_prompt=initial_prompt,
    )
    stage._ensure_model()
    seg = DiarizedSegment(
        t0=0.0,
        t1=pcm.shape[0] / float(sr),
        sample_rate=sr,
        speaker="S0",
        pcm=pcm.astype(np.float32, copy=False),
    )
    utt = stage._transcribe_blocking(seg)
    return "" if utt is None else utt["text"]
