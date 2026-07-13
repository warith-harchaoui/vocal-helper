"""
vocal_helper.eot
================

Semantic end-of-turn (EOT) detection stage — inspired by LiveKit's
``turn-detector`` model (April 2026 release, see
``livekit/turn-detector`` on Hugging Face) which fine-tuned a
Qwen2.5-0.5B distilled from Qwen2.5-7B to score, in ~10 ms / inference,
whether a partial transcript looks like a *completed* speaker turn.

LiveKit reports a 39 % reduction in false-positive interruptions when
the semantic EOT signal is fused with Silero VAD. Their claim is that
the latency cost of turn-detection is the largest hidden contributor
to perceived voice-agent lag.

Why this matters for vocal-helper
---------------------------------
Our :class:`SileroVADStage` emits a :class:`VoicedSegment` after
``min_silence_ms`` of trailing silence. That's a rigid threshold : a
speaker who takes a 350 ms breath mid-sentence gets cut into two
segments, the ASR sees two fragments, the diarizer sees two
embeddings (sometimes assigning different speakers !), and the LLM
analyst gets a worse signal. Empirically the AMI dev-slice contains
~ 15 % of utterances under 400 ms — they are mostly back-channels and
breaths, not closed turns.

The :class:`SemanticEOTStage` sits between :class:`SileroVADStage`
and :class:`OnlineDiarStage`. For every incoming :class:`VoicedSegment`
it :

1. Runs a fast STT pass (whisper.cpp turbo, same model the downstream
   :class:`WhisperStage` uses — kept in a thread pool to avoid
   stalling the loop).
2. Asks a small classifier LLM (``qwen2.5:3b`` by default — close
   enough in capability to LiveKit's distilled 0.5B target, already
   available via Ollama on the user's machine) whether the partial
   transcript is a complete thought.
3. If complete → emit the segment immediately.
4. If incomplete → buffer it, wait for the next segment, then merge
   and re-evaluate. After ``max_merge_s`` seconds of accumulation
   we force an emit regardless.

The stage is :class:`opt-in` — disabled by default to keep the
zero-dependency path (Silero alone) ; users wire it in via
:class:`PipelineConfig.eot`.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vocal_helper.types import VoicedSegment

DEFAULT_EOT_MODEL = "qwen2.5:3b"
DEFAULT_STT_MODEL = "large-v3-turbo-q5_0"
DEFAULT_MAX_MERGE_S = 4.0
DEFAULT_MIN_INCOMPLETE_MS = 800

# Compact yes/no prompt — shorter generations = faster classification.
_PROMPT = (
    "You are a speech end-of-turn classifier. Given the latest snippet "
    "of a single speaker's utterance, answer with exactly one word :\n"
    " - YES if the utterance looks like a complete turn (the speaker "
    "is done and could plausibly hand the floor over).\n"
    " - NO if it ends mid-thought, mid-clause, mid-word, or with a "
    "filler that signals the speaker is about to continue.\n\n"
    "Utterance: {text}\n\nAnswer:"
)


@dataclass
class _PendingSegment:
    """One in-flight VoicedSegment held back pending a follow-up."""

    seg: VoicedSegment
    received_at: float
    accumulated_text: str


class SemanticEOTStage:
    """Producer/consumer EOT gating stage.

    Parameters
    ----------
    eot_model : str
        Ollama model used as the EOT classifier. Default ``qwen2.5:3b``
        — small enough to run at ~ 50 ms / classification on Apple
        Silicon while broadly equivalent in capability to the LiveKit
        turn-detector's 0.5B target.
    stt_model : str
        pywhispercpp model used for the partial transcript pass.
        Default ``large-v3-turbo-q5_0`` — same as the downstream
        :class:`WhisperStage`. We could cache one instance shared by
        both stages in a future revision.
    max_merge_s : float
        Maximum total duration of a merged-on-incomplete chain. After
        this we force-emit regardless of the classifier's verdict.
    min_incomplete_ms : int
        Segments shorter than this are presumed back-channels (acks /
        breaths) and gated by the classifier ; longer segments are
        emitted directly without an LLM call (cheap heuristic).
    host : str, optional
        Ollama host URL. Defaults to the ``OLLAMA_HOST`` env var or
        ``http://localhost:11434``.
    """

    def __init__(
        self,
        *,
        eot_model: str = DEFAULT_EOT_MODEL,
        stt_model: str = DEFAULT_STT_MODEL,
        max_merge_s: float = DEFAULT_MAX_MERGE_S,
        min_incomplete_ms: int = DEFAULT_MIN_INCOMPLETE_MS,
        host: str | None = None,
    ) -> None:
        self.eot_model = eot_model
        self.stt_model = stt_model
        self.max_merge_s = max_merge_s
        self.min_incomplete_ms = min_incomplete_ms
        self.host = host
        self._ollama = None
        self._whisper = None
        self._pending: _PendingSegment | None = None

    # ----- lifecycle ------------------------------------------------------

    def _ensure_clients(self) -> None:
        if self._ollama is None:
            try:
                import ollama  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "SemanticEOTStage requires ollama. "
                    "Install with `pip install vocal-helper[llm]`."
                ) from e
            self._ollama = ollama.Client(host=self.host) if self.host else ollama.Client()
        if self._whisper is None:
            try:
                from pywhispercpp.model import Model  # type: ignore
            except ImportError as e:
                raise ImportError("SemanticEOTStage requires pywhispercpp.") from e
            self._whisper = Model(
                self.stt_model,
                n_threads=6,
                print_realtime=False,
                print_progress=False,
            )

    # ----- public coroutine ----------------------------------------------

    async def run(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
    ) -> None:
        """Consume :class:`VoicedSegment`s, gate them by semantic EOT."""
        self._ensure_clients()
        while True:
            item = await inbox.get()
            if item is None:
                # Flush any pending segment on shutdown.
                if self._pending is not None:
                    await outbox.put(self._pending.seg)
                    self._pending = None
                await outbox.put(None)
                return
            decisions = await self._handle(item)
            for seg in decisions:
                await outbox.put(seg)

    # ----- core ---------------------------------------------------------

    async def _handle(self, seg: VoicedSegment) -> list[VoicedSegment]:
        dur_ms = (seg["t1"] - seg["t0"]) * 1000.0

        # Short segments — gate by classifier.
        if self._pending is None and dur_ms >= self.min_incomplete_ms:
            # Long enough that it's almost certainly a complete turn —
            # skip the classifier call and emit directly.
            return [seg]

        # Build the candidate (either fresh or merged-with-pending).
        if self._pending is None:
            candidate = seg
            accumulated_text = ""
        else:
            candidate = self._merge_segments(self._pending.seg, seg)
            accumulated_text = self._pending.accumulated_text

        # Time-cap force-emit.
        candidate_dur = candidate["t1"] - candidate["t0"]
        if candidate_dur >= self.max_merge_s:
            self._pending = None
            return [candidate]

        # Get partial transcript + classify.
        text = await asyncio.to_thread(self._partial_transcribe, candidate["pcm"])
        full_text = (accumulated_text + " " + text).strip()
        complete = await asyncio.to_thread(self._classify, full_text)

        if complete:
            self._pending = None
            return [candidate]

        # Hold back ; wait for the next segment to merge.
        self._pending = _PendingSegment(
            seg=candidate,
            received_at=time.monotonic(),
            accumulated_text=full_text,
        )
        return []

    # ----- helpers ------------------------------------------------------

    def _merge_segments(self, a: VoicedSegment, b: VoicedSegment) -> VoicedSegment:
        """Concatenate two VoicedSegments preserving the parent time."""
        gap_samples = max(0, int(round((b["t0"] - a["t1"]) * a["sample_rate"])))
        gap = np.zeros(gap_samples, dtype=np.float32) if gap_samples else None
        parts = [a["pcm"]]
        if gap is not None:
            parts.append(gap)
        parts.append(b["pcm"])
        return VoicedSegment(
            t0=a["t0"],
            t1=b["t1"],
            sample_rate=a["sample_rate"],
            pcm=np.concatenate(parts, axis=0),
        )

    def _partial_transcribe(self, pcm: NDArray[np.float32]) -> str:
        assert self._whisper is not None
        try:
            segs = self._whisper.transcribe(pcm)
        except Exception:  # noqa: BLE001
            return ""
        return " ".join((s.text or "").strip() for s in segs).strip()

    def _classify(self, text: str) -> bool:
        """Ask the EOT classifier ; return True iff utterance looks complete."""
        if not text.strip():
            return True  # nothing to extend — emit
        prompt = _PROMPT.format(text=text)
        try:
            resp = self._ollama.generate(model=self.eot_model, prompt=prompt, stream=False)
        except Exception:  # noqa: BLE001
            return True  # classifier offline → fall back to non-gated VAD behaviour
        if isinstance(resp, dict):
            answer = str(resp.get("response", "")).strip().lower()
        else:
            answer = str(getattr(resp, "response", "")).strip().lower()
        # Liberal parser : look for YES somewhere in the first ~ 10 chars.
        head = answer[:10]
        return "yes" in head and "no" not in head[: head.find("yes") + 3]
