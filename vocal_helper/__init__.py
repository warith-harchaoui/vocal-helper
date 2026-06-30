"""
Vocal Helper
============

Producer/consumer pipeline turning a live PCM stream into diarized,
transcribed utterances and (optionally) a rolling LLM summary.

Stages, all stitched by :class:`Pipeline` :

1. **Source** — any async iterator of :class:`PcmFrame`. Three
   shipped : :func:`sources.from_microphone` (live mic via
   ``capture-helper``), :func:`sources.from_wav_file` (replay a
   mono 16 kHz WAV at real-time or burst speed),
   :func:`sources.from_numpy_array` (in-memory PCM, for tests).
2. **VAD** — Silero v5 ONNX on CPU (:class:`SileroVADStage`).
   32 ms windows, ``activity_threshold=0.5``, default
   ``min_silence_ms=300``.
3. **Online diarization** — :class:`OnlineDiarStage`. Per-segment
   embedding (pyannote/embedding by default, TitaNet via NeMo with
   ``backend='nemo'``) + cosine-distance running-mean clustering
   with ``join_threshold=0.30`` (calibrated on AMI dev-slice N=8,
   2026-06-30 sweep).
4. **STT** — :class:`WhisperStage`. pywhispercpp turbo
   (``large-v3-turbo-q5_0``), threads default 6, word timestamps on.
   Runs in :func:`asyncio.to_thread` so the loop is never blocked.
5. **LLM analyst** (optional) — :class:`GemmaAnalystStage`. Ollama
   serves ``gemma4:e4b`` (auto-selects the ``-mlx`` variant on
   Apple-Silicon) ; the stage keeps a rolling summary of everything
   older than ``recent_window_s = 60`` seconds and emits a fresh
   :class:`SummarySnapshot` after every accepted utterance.

Quickstart
----------

>>> import asyncio, vocal_helper as vh
>>>
>>> async def main():
...     pipeline = vh.Pipeline(
...         source=lambda: vh.sources.from_microphone(),
...         config=vh.PipelineConfig(
...             diar={"backend": "pyannote"},
...             llm={"model": "gemma4:e4b"},
...         ),
...     )
...     async for ev in pipeline.run():
...         if isinstance(ev, dict) and "text" in ev:
...             print(f"[{ev['t0']:.1f}s {ev['speaker']}] {ev['text']}")
...         elif isinstance(ev, dict) and "summary" in ev:
...             print(f"--- summary ---\\n{ev['summary']}")
>>>
>>> asyncio.run(main())

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from vocal_helper import sources
from vocal_helper.asr import WhisperStage, transcribe_pcm
from vocal_helper.diar import OfflineDiarStage, OnlineDiarStage
from vocal_helper.llm import GemmaAnalystStage
from vocal_helper.pipeline import (
    OfflinePipeline,
    OfflinePipelineConfig,
    Pipeline,
    PipelineConfig,
)
from vocal_helper.types import (
    DiarizedSegment,
    PcmFrame,
    SummarySnapshot,
    Utterance,
    VoicedSegment,
)
from vocal_helper.vad import SileroVADStage

__all__ = [
    "sources",
    "Pipeline",
    "PipelineConfig",
    "OfflinePipeline",
    "OfflinePipelineConfig",
    "SileroVADStage",
    "OnlineDiarStage",
    "OfflineDiarStage",
    "WhisperStage",
    "GemmaAnalystStage",
    "PcmFrame",
    "VoicedSegment",
    "DiarizedSegment",
    "Utterance",
    "SummarySnapshot",
    "transcribe_pcm",
]

__version__ = "0.1.0"
