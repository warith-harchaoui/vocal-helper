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

>>> import asyncio, vocal_helper as voh
>>>
>>> async def main():
...     pipeline = voh.Pipeline(
...         source=lambda: voh.sources.from_microphone(),
...         config=voh.PipelineConfig(
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

Usage Example
-------------
>>> import asyncio, vocal_helper as voh
>>>
>>> async def main():
...     pipeline = voh.Pipeline(
...         source=lambda: voh.sources.from_wav_file("clip.wav"),
...         config=voh.PipelineConfig(diar={"backend": "pyannote"}),
...     )
...     async for ev in pipeline.run():
...         if "text" in ev:
...             print(ev["text"])
>>>
>>> asyncio.run(main())

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from vocal_helper import sources
from vocal_helper.asr import WhisperStage, transcribe_pcm
from vocal_helper.diar import OfflineDiarStage, OnlineDiarStage
from vocal_helper.lid import (
    LangRegion,
    RegionVerdict,
    cross_check_regions,
    detect_language,
    detect_language_regions,
    detect_language_regions_fast,
    detect_language_speechbrain,
    language_posterior_curve,
)
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

# Optional in-flight modules — imported best-effort so the base package
# stays importable even when the WIP EOT / parallel-pipelines / TTS
# modules are absent (they live behind ``git stash`` while the
# multi-surface upgrade lands).
try:
    from vocal_helper.eot import SemanticEOTStage  # type: ignore[assignment]
except Exception:  # pragma: no cover — optional
    SemanticEOTStage = None  # type: ignore[assignment]

try:
    from vocal_helper.eot_bench import (  # type: ignore[assignment]
        EOTPair,
        false_cutoff_rate,
        hang_rate,
    )
    from vocal_helper.eot_bench import score as eot_score  # type: ignore[assignment]
except Exception:  # pragma: no cover — optional
    EOTPair = None  # type: ignore[assignment]
    false_cutoff_rate = None  # type: ignore[assignment]
    hang_rate = None  # type: ignore[assignment]
    eot_score = None  # type: ignore[assignment]

try:
    from vocal_helper.parallel_pipelines import (  # type: ignore[assignment]
        run_parallel_async,
        run_parallel_sync,
    )
except Exception:  # pragma: no cover — optional
    run_parallel_async = None  # type: ignore[assignment]
    run_parallel_sync = None  # type: ignore[assignment]

__all__ = [
    "sources",
    "Pipeline",
    "PipelineConfig",
    "OfflinePipeline",
    "OfflinePipelineConfig",
    "SileroVADStage",
    "OnlineDiarStage",
    "OfflineDiarStage",
    "SemanticEOTStage",
    "WhisperStage",
    # Spoken-language diarization (see vocal_helper.lid).
    "LangRegion",
    "RegionVerdict",
    "cross_check_regions",
    "detect_language",
    "detect_language_regions",
    "detect_language_regions_fast",
    "detect_language_speechbrain",
    "language_posterior_curve",
    "GemmaAnalystStage",
    # LiveKit-inspired EOT eval + Pipecat-inspired parallel primitive
    "EOTPair",
    "eot_score",
    "false_cutoff_rate",
    "hang_rate",
    "run_parallel_sync",
    "run_parallel_async",
    "PcmFrame",
    "VoicedSegment",
    "DiarizedSegment",
    "Utterance",
    "SummarySnapshot",
    "transcribe_pcm",
]

__author__ = "Warith Harchaoui, Ph.D."
__email__ = "warithmetics@deraison.ai"
__version__ = "0.4.2"
