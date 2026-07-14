"""Tests for the :class:`Pipeline` / :class:`OfflinePipeline` glue.

We don't run end-to-end here — that would pull in pyannote, whisper
and ollama (the ``integration`` marker covers that). What we do
check :

* Config dataclasses honour their defaults.
* Constructors size the internal queues from the config.
* Subscribers register without side effects.
* The ``stage`` kwargs pre-validate cheaply (no models loaded).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import vocal_helper as vh

# ---------------------------------------------------------------------------
# PipelineConfig defaults
# ---------------------------------------------------------------------------


def test_pipeline_config_defaults_are_independent() -> None:
    """Two configs must NOT share the per-stage default dicts."""
    a = vh.PipelineConfig()
    b = vh.PipelineConfig()
    a.diar["backend"] = "nemo"
    assert b.diar == {}, "PipelineConfig leaked a shared diar dict"


def test_pipeline_config_defaults_values() -> None:
    cfg = vh.PipelineConfig()
    assert cfg.qsize_pcm > 0
    assert cfg.qsize_seg > 0
    assert cfg.vad == {}
    assert cfg.diar == {}
    assert cfg.asr == {}
    # Optional stages default to ``None`` — enable by passing a dict.
    assert cfg.eot is None, (
        "SemanticEOTStage must be opt-in — one extra LLM hop per voiced segment is not free"
    )
    assert cfg.llm is None


def test_pipeline_config_eot_accepts_dict() -> None:
    """When callers opt into semantic EOT, the config must round-trip the dict."""
    cfg = vh.PipelineConfig(eot={"model": "gemma4:e4b", "threshold": 0.5})
    assert cfg.eot == {"model": "gemma4:e4b", "threshold": 0.5}


def test_offline_pipeline_config_defaults() -> None:
    cfg = vh.OfflinePipelineConfig()
    assert cfg.qsize_pcm > 0
    assert cfg.qsize_seg > 0
    assert cfg.llm is None
    # No VAD block — the offline path delegates to the diar backend.
    assert not hasattr(cfg, "vad")
    # No EOT block either — semantic EOT only makes sense on a live
    # stream where cutting a caller off mid-sentence matters.
    assert not hasattr(cfg, "eot")


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _silent_source(duration_s: float = 0.05):
    """Build a source factory returning ``duration_s`` of silence."""
    n = int(16_000 * duration_s)
    pcm = np.zeros(n, dtype=np.float32)
    return lambda: vh.sources.from_numpy_array(pcm)


def test_pipeline_queues_honour_config_sizes() -> None:
    cfg = vh.PipelineConfig(qsize_pcm=7, qsize_seg=3)
    p = vh.Pipeline(source=_silent_source(), config=cfg)
    # Queue sizes are private but we can probe via the public attr.
    assert p._q_pcm.maxsize == 7
    assert p._q_voiced.maxsize == 3
    assert p._q_diar.maxsize == 3
    assert p._q_utt.maxsize == 3
    assert p._q_summary.maxsize == 3


def test_pipeline_disables_llm_when_unconfigured() -> None:
    cfg = vh.PipelineConfig()  # llm=None
    p = vh.Pipeline(source=_silent_source(), config=cfg)
    assert p._llm is None


def test_pipeline_subscribers_register() -> None:
    p = vh.Pipeline(source=_silent_source())
    seen: list[str] = []

    async def voiced_cb(_x: object) -> None:
        seen.append("voiced")

    async def diar_cb(_x: object) -> None:
        seen.append("diar")

    async def utt_cb(_x: object) -> None:
        seen.append("utt")

    p.subscribe_voiced(voiced_cb)
    p.subscribe_diarized(diar_cb)
    p.subscribe_utterances(utt_cb)
    assert len(p._voiced_subs) == 1
    assert len(p._diar_subs) == 1
    assert len(p._utt_subs) == 1


# ---------------------------------------------------------------------------
# OnlineDiarStage construction-time validation
# ---------------------------------------------------------------------------


def test_online_diar_rejects_bad_join_threshold() -> None:
    """Out-of-range thresholds must blow up at construction, not runtime."""
    from vocal_helper.diar import OnlineDiarStage

    with pytest.raises(ValueError, match="join_threshold"):
        OnlineDiarStage(join_threshold=-0.1)
    with pytest.raises(ValueError, match="join_threshold"):
        OnlineDiarStage(join_threshold=2.5)


def test_online_diar_rejects_bad_ema_alpha() -> None:
    from vocal_helper.diar import OnlineDiarStage

    with pytest.raises(ValueError, match="ema_alpha"):
        OnlineDiarStage(ema_alpha=0.0)
    with pytest.raises(ValueError, match="ema_alpha"):
        OnlineDiarStage(ema_alpha=1.5)


# ---------------------------------------------------------------------------
# Pipeline shutdown — no source means no events, but ``run`` should still
# complete cleanly when the source is exhausted. We *don't* load real
# stages here — they'd try to fetch models. Instead we check that the
# event loop wraps up via cancellation on an empty source.
# ---------------------------------------------------------------------------


def test_pipeline_run_completes_with_empty_buffer() -> None:
    """Empty PCM buffer → ``None`` sentinel cascades, ``run`` ends without yielding."""
    # Patch the heavy stages with no-op coroutines so this stays unit-fast.
    pcm = np.zeros(0, dtype=np.float32)

    p = vh.Pipeline(
        source=lambda: vh.sources.from_numpy_array(pcm),
        config=vh.PipelineConfig(),
    )

    async def passthrough_vad(inbox: asyncio.Queue, outbox: asyncio.Queue) -> None:
        while True:
            item = await inbox.get()
            if item is None:
                await outbox.put(None)
                return

    async def passthrough_diar(inbox: asyncio.Queue, outbox: asyncio.Queue) -> None:
        await passthrough_vad(inbox, outbox)

    async def passthrough_asr(inbox: asyncio.Queue, outbox: asyncio.Queue) -> None:
        await passthrough_vad(inbox, outbox)

    p._vad.run = passthrough_vad  # type: ignore[assignment]
    p._diar.run = passthrough_diar  # type: ignore[assignment]
    p._asr.run = passthrough_asr  # type: ignore[assignment]

    async def drive() -> list:
        return [ev async for ev in p.run()]

    events = asyncio.run(drive())
    assert events == []
