"""Smoke tests — no model loading, no network."""

from __future__ import annotations

import asyncio

import numpy as np

import vocal_helper as vh


def test_imports() -> None:
    """Every public symbol should be importable without optional deps."""
    for name in vh.__all__:
        assert hasattr(vh, name), name


def test_pcm_frame_shape() -> None:
    """``from_numpy_array`` should chunk a buffer at the configured frame size."""

    async def collect() -> list[vh.PcmFrame]:
        pcm = np.zeros(16_000, dtype=np.float32)  # 1 s @ 16 kHz
        out: list[vh.PcmFrame] = []
        async for f in vh.sources.from_numpy_array(pcm, sample_rate=16_000, frame_ms=20):
            out.append(f)
        return out

    frames = asyncio.run(collect())
    # 1000 / 20 = 50 frames.
    assert len(frames) == 50
    assert frames[0]["pcm"].shape == (320,)
    assert frames[0]["sample_rate"] == 16_000
    # t0 should be monotonically increasing.
    times = [f["t0"] for f in frames]
    assert times == sorted(times)


def test_pipeline_config_defaults() -> None:
    """Defaults shouldn't blow up at construction."""
    cfg = vh.PipelineConfig()
    assert cfg.qsize_pcm > 0
    assert cfg.qsize_seg > 0
    assert cfg.llm is None
    # PCM-only source can build a Pipeline — we don't run it here so no
    # models are loaded.
    pcm = np.zeros(800, dtype=np.float32)
    pipeline = vh.Pipeline(
        source=lambda: vh.sources.from_numpy_array(pcm),
        config=cfg,
    )
    assert pipeline is not None
