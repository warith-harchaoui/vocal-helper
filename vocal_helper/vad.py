"""
vocal_helper.vad
================

Silero-VAD stage. Consumes :class:`PcmFrame` and emits one
:class:`VoicedSegment` per voiced run.

Algorithm
---------
- Maintain a rolling buffer.
- Score every 512-sample (~32 ms @ 16 kHz) Silero window ; threshold
  at ``activity_threshold`` (default 0.5).
- A speech run starts the first time the score crosses upward, and
  ends after ``min_silence_ms`` of trailing silence below threshold.
- The emitted PCM is the entire voiced span padded with a small
  lead/trail margin so the ASR sees the natural envelope (default
  ``edge_pad_ms = 200``).
- Runs shorter than ``min_speech_ms`` (default 300 ms) are dropped
  — usually finger taps, lip smacks, room noise.

These thresholds match the canonical pdbms settings ; see
``pdbms.utils.snr`` and ``vad-cadence-study.md`` §10 for the
operating-point justification.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vocal_helper.types import PcmFrame, VoicedSegment

# 32 ms @ 16 kHz — Silero v5 native window.
_SILERO_WINDOW_SAMPLES = 512


@dataclass
class _Run:
    """One in-progress voiced run."""

    t0: float
    samples: list[NDArray[np.float32]]
    last_voiced_sample: int  # absolute sample index of last voiced window
    last_window_sample: int  # absolute sample index of last scored window


class SileroVADStage:
    """Producer/consumer Silero-VAD stage.

    Parameters
    ----------
    activity_threshold : float
        Silero score above which a window is considered voiced.
        Default 0.5 (canonical).
    min_silence_ms : int
        Trailing silence required to close a voiced run. Default
        300 ms.
    min_speech_ms : int
        Reject runs shorter than this. Default 300 ms.
    edge_pad_ms : int
        Lead / trail padding kept around the voiced span so the
        ASR sees a natural envelope. Default 200 ms.
    sample_rate : int
        Required to be 16 000 — Silero v5 is trained at 16 kHz.

    Notes
    -----
    The stage owns a single Silero model instance (loaded lazily on
    first frame). The model is CPU-only ONNX so this is cheap.
    """

    def __init__(
        self,
        *,
        activity_threshold: float = 0.5,
        min_silence_ms: int = 300,
        min_speech_ms: int = 300,
        edge_pad_ms: int = 200,
        sample_rate: int = 16_000,
    ) -> None:
        """Configure the Silero VAD stage ; the model loads lazily.

        Parameters
        ----------
        activity_threshold : float
            Silero speech-probability threshold above which a window
            counts as voiced. Default 0.5.
        min_silence_ms : int
            Minimum silence duration that closes an ongoing run.
            Default 300 ms.
        min_speech_ms : int
            Minimum voiced duration for a run to be emitted at all.
            Default 300 ms.
        edge_pad_ms : int
            Context padding prepended / appended to each emitted run so
            ASR sees a natural envelope. Default 200 ms.
        sample_rate : int
            Input sample rate ; must be 16 000 — Silero v5 is trained at
            16 kHz. Default 16 000.

        Raises
        ------
        ValueError
            If ``sample_rate`` is not 16 000.
        """
        if sample_rate != 16_000:
            raise ValueError(f"SileroVADStage requires sample_rate=16000, got {sample_rate}")
        self.activity_threshold = activity_threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.edge_pad_ms = edge_pad_ms
        self.sample_rate = sample_rate

        self._model: Any = None
        self._torch: Any = None
        # Carry-over leftover at every frame boundary so a Silero
        # window is never split across two frames.
        self._tail = np.zeros(0, dtype=np.float32)
        # Absolute sample counter — used to map runs back to seconds.
        self._absolute_sample = 0
        self._run: _Run | None = None
        # Tiny ring of unvoiced tail frames so we can prepend
        # ``edge_pad_ms`` of context to each emitted run.
        edge_pad_samples = sample_rate * edge_pad_ms // 1000
        self._lead_pad: deque[NDArray[np.float32]] = deque(
            maxlen=max(1, edge_pad_samples // _SILERO_WINDOW_SAMPLES)
        )

    # ----- lifecycle ------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily load the Silero VAD model (and torch handle) on first use.

        Idempotent — returns immediately once the model is loaded, so it
        is safe to call at the top of :meth:`run`. The CPU-only ONNX model
        makes this cheap.
        """
        if self._model is not None:
            return
        import torch  # type: ignore
        from silero_vad import load_silero_vad  # type: ignore

        self._torch = torch
        self._model = load_silero_vad()

    # ----- public coroutine ----------------------------------------------

    async def run(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
    ) -> None:
        """Consume PcmFrames from ``inbox``, push VoicedSegments to ``outbox``.

        Exits cleanly when a ``None`` sentinel is received. Forwards
        ``None`` to ``outbox`` so downstream stages can stop too.
        """
        self._ensure_model()
        while True:
            item = await inbox.get()
            if item is None:
                # Close any in-flight run before signalling end-of-stream.
                if self._run is not None:
                    seg = self._emit_run(self._run, end_t=self._absolute_sample)
                    self._run = None
                    if seg is not None:
                        await outbox.put(seg)
                await outbox.put(None)
                return
            segs = self._consume_frame(item)
            for s in segs:
                await outbox.put(s)

    # ----- the actual VAD logic ------------------------------------------

    def _consume_frame(self, frame: PcmFrame) -> list[VoicedSegment]:
        """Process one inbound PCM frame ; return zero or more emitted runs."""
        pcm = frame["pcm"]
        # Concatenate tail + new frame so window borders are stable.
        buf = np.concatenate([self._tail, pcm], axis=0)
        n = buf.shape[0]
        emitted: list[VoicedSegment] = []
        cursor = 0
        while cursor + _SILERO_WINDOW_SAMPLES <= n:
            window = buf[cursor : cursor + _SILERO_WINDOW_SAMPLES]
            score = self._score_window(window)
            voiced = score >= self.activity_threshold
            abs_start = self._absolute_sample + cursor
            if voiced:
                if self._run is None:
                    self._run = _Run(
                        t0=max(
                            0.0, abs_start / float(self.sample_rate) - self.edge_pad_ms / 1000.0
                        ),
                        samples=list(self._lead_pad) + [window.copy()],
                        last_voiced_sample=abs_start + _SILERO_WINDOW_SAMPLES,
                        last_window_sample=abs_start + _SILERO_WINDOW_SAMPLES,
                    )
                else:
                    self._run.samples.append(window.copy())
                    self._run.last_voiced_sample = abs_start + _SILERO_WINDOW_SAMPLES
                    self._run.last_window_sample = abs_start + _SILERO_WINDOW_SAMPLES
            else:
                if self._run is None:
                    # Idle — feed the lead pad ring so the next run starts
                    # with a natural envelope.
                    self._lead_pad.append(window.copy())
                else:
                    # In-progress run — keep accumulating until silence
                    # window crosses ``min_silence_ms``.
                    self._run.samples.append(window.copy())
                    self._run.last_window_sample = abs_start + _SILERO_WINDOW_SAMPLES
                    silence_ms = (
                        (self._run.last_window_sample - self._run.last_voiced_sample)
                        * 1000.0
                        / self.sample_rate
                    )
                    if silence_ms >= self.min_silence_ms:
                        seg = self._emit_run(self._run, end_t=self._run.last_voiced_sample)
                        self._run = None
                        if seg is not None:
                            emitted.append(seg)
                        # Reset the lead pad with the just-scored unvoiced
                        # window so a *new* run can prepend it.
                        self._lead_pad.clear()
                        self._lead_pad.append(window.copy())
            cursor += _SILERO_WINDOW_SAMPLES
        # Stash the leftover < 512 samples for the next frame.
        self._tail = buf[cursor:].copy()
        self._absolute_sample += pcm.shape[0]
        return emitted

    def _score_window(self, window: NDArray[np.float32]) -> float:
        """Run one Silero forward pass on a 512-sample window."""
        assert self._torch is not None
        with self._torch.no_grad():
            return float(self._model(self._torch.from_numpy(window), self.sample_rate).item())

    def _emit_run(self, run: _Run, end_t: int) -> VoicedSegment | None:
        """Materialise a finished voiced run into a :class:`VoicedSegment`.

        ``end_t`` is the absolute sample index of the run's last
        voiced window — we trim the trailing pure-silence pad to
        ``edge_pad_ms``.
        """
        cat = np.concatenate(run.samples, axis=0)
        # Compute t1 from the last-voiced index plus the configured edge pad.
        t1_samples = end_t + self.sample_rate * self.edge_pad_ms // 1000
        t1 = t1_samples / float(self.sample_rate)
        duration_ms = (t1 - run.t0) * 1000.0
        # Drop sub-threshold runs.
        if duration_ms < self.min_speech_ms:
            return None
        # Cap cat to the t1 window so we don't carry past-min-silence noise.
        max_samples = int(round((t1 - run.t0) * self.sample_rate))
        if cat.shape[0] > max_samples:
            cat = cat[:max_samples]
        return VoicedSegment(
            t0=run.t0,
            t1=t1,
            sample_rate=self.sample_rate,
            pcm=cat.astype(np.float32, copy=False),
        )
