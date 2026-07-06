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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vocal_helper.types import DiarizedSegment, Utterance

DEFAULT_MODEL = "large-v3-turbo-q5_0"
DEFAULT_LANGUAGE = "auto"
DEFAULT_THREADS = 6
DEFAULT_MIN_SEGMENT_MS = 250
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
    ) -> None:
        self.model_name = model
        self.language = language
        self.threads = threads
        self.word_timestamps = word_timestamps
        self.initial_prompt = initial_prompt
        self.min_segment_ms = min_segment_ms
        self._model: Any | None = None

    # ----- lifecycle ------------------------------------------------------

    def _ensure_model(self) -> None:
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

    # ----- public coroutine ----------------------------------------------

    async def run(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
    ) -> None:
        """Consume :class:`DiarizedSegment` from ``inbox``, push :class:`Utterance`."""
        self._ensure_model()
        while True:
            item = await inbox.get()
            if item is None:
                await outbox.put(None)
                return
            utt = await asyncio.to_thread(self._transcribe_blocking, item)
            if utt is not None:
                await outbox.put(utt)

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
                    seg["pcm"], initial_prompt=self.initial_prompt,
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
) -> str:
    """Synchronous one-shot transcription. Loads whisper.cpp on call."""
    stage = WhisperStage(
        model=model, language=language, threads=threads, word_timestamps=False,
    )
    stage._ensure_model()
    seg = DiarizedSegment(
        t0=0.0, t1=pcm.shape[0] / float(sr),
        sample_rate=sr, speaker="S0",
        pcm=pcm.astype(np.float32, copy=False),
    )
    utt = stage._transcribe_blocking(seg)
    return "" if utt is None else utt["text"]
