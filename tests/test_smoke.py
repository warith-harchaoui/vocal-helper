"""Smoke tests — no model loading, no network."""

from __future__ import annotations

import asyncio

import numpy as np

import vocal_helper as voh


def test_public_imports_and_pipeline_construction() -> None:
    """Every public symbol imports and a PCM-only Pipeline builds without loading models.

    A single no-deps construction smoke: every name in ``__all__`` resolves off
    the top-level package, ``PipelineConfig`` defaults are sane (positive queue
    sizes, no LLM), and a ``Pipeline`` built from a pure-PCM source constructs
    without touching any heavy stage (we never run it, so no models load).
    """
    for name in voh.__all__:
        assert hasattr(voh, name), name  # importable without optional deps

    cfg = voh.PipelineConfig()
    assert cfg.qsize_pcm > 0
    assert cfg.qsize_seg > 0
    assert cfg.llm is None  # no analyst wired by default

    # A PCM-only source is enough to construct a Pipeline; construction alone
    # must not load Whisper / diarization / VAD models.
    pcm = np.zeros(800, dtype=np.float32)
    pipeline = voh.Pipeline(
        source=lambda: voh.sources.from_numpy_array(pcm),
        config=cfg,
    )
    assert pipeline is not None


def test_pcm_frame_shape() -> None:
    """``from_numpy_array`` chunks a buffer into monotonic fixed-size frames.

    Drives the async source generator over 1 s of silence and asserts the
    framing contract: 20 ms frames at 16 kHz yield 50 frames of 320 samples,
    each tagged with the source sample rate and a monotonically increasing
    ``t0`` timestamp.
    """

    async def collect() -> list[voh.PcmFrame]:
        """Chunk 1 s of silence into frames and gather them into a list.

        Returns
        -------
        list[voh.PcmFrame]
            All frames emitted by the source generator.
        """
        pcm = np.zeros(16_000, dtype=np.float32)  # 1 s @ 16 kHz
        out: list[voh.PcmFrame] = []
        async for f in voh.sources.from_numpy_array(pcm, sample_rate=16_000, frame_ms=20):
            out.append(f)
        return out

    frames = asyncio.run(collect())
    assert len(frames) == 50  # 1000 ms / 20 ms
    assert frames[0]["pcm"].shape == (320,)  # 16 kHz * 20 ms
    assert frames[0]["sample_rate"] == 16_000
    times = [f["t0"] for f in frames]
    assert times == sorted(times)  # t0 monotonically increasing
