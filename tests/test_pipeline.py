"""Tests for the :class:`Pipeline` / :class:`OfflinePipeline` glue.

We don't run end-to-end with real models here — that would pull in
pyannote, whisper and ollama (the ``integration`` marker covers that).
What we do check :

* Config dataclasses honour their defaults and don't leak shared state.
* Constructors size the internal queues from the config and wire
  subscribers / optional stages correctly.
* Construction-time validation rejects bad stage kwargs cheaply.
* The event loop drains cleanly when the source is exhausted — using
  no-op stage stubs so no model is ever loaded.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import vocal_helper as voh

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


def test_pipeline_config_defaults_and_eot_opt_in() -> None:
    """``PipelineConfig`` defaults are independent, sane, and EOT is opt-in.

    Folds the config contract into one flow: two instances must not share
    the per-stage default dicts (mutating one leaves the other empty), the
    documented defaults hold (positive queue sizes, empty stage dicts,
    optional ``eot`` / ``llm`` off), and opting into semantic EOT round-trips
    the supplied dict verbatim.
    """
    # Independence: the per-stage dicts are per-instance, not class-shared.
    a = voh.PipelineConfig()
    b = voh.PipelineConfig()
    a.diar["backend"] = "nemo"
    assert b.diar == {}, "PipelineConfig leaked a shared diar dict"

    # Documented defaults: queues sized, mandatory stages empty-configured.
    cfg = voh.PipelineConfig()
    assert cfg.qsize_pcm > 0
    assert cfg.qsize_seg > 0
    assert cfg.vad == {}
    assert cfg.diar == {}
    assert cfg.asr == {}
    # Optional stages default off — an extra LLM hop per segment is not free.
    assert cfg.eot is None, "SemanticEOTStage must be opt-in"
    assert cfg.llm is None

    # Opt-in: a supplied EOT block round-trips unchanged.
    opted = voh.PipelineConfig(eot={"model": "gemma4:e4b", "threshold": 0.5})
    assert opted.eot == {"model": "gemma4:e4b", "threshold": 0.5}


def test_offline_pipeline_config_defaults() -> None:
    """``OfflinePipelineConfig`` omits the VAD and EOT blocks the live path carries.

    The offline path delegates voicing to the diar backend (no VAD block)
    and has no live stream to cut off mid-sentence (no EOT block), while
    still sizing its queues and defaulting the analyst off.
    """
    cfg = voh.OfflinePipelineConfig()
    assert cfg.qsize_pcm > 0
    assert cfg.qsize_seg > 0
    assert cfg.llm is None
    # No VAD block — the offline path delegates to the diar backend.
    assert not hasattr(cfg, "vad")
    # No EOT block — semantic EOT only matters on a live stream.
    assert not hasattr(cfg, "eot")


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _silent_source(duration_s: float = 0.05):
    """Build a source factory returning ``duration_s`` of silence.

    Parameters
    ----------
    duration_s : float, optional
        Length of the silent buffer in seconds (default 0.05).

    Returns
    -------
    Callable[[], AsyncIterator[vocal_helper.PcmFrame]]
        Zero-arg factory yielding a fresh silent numpy source.
    """
    n = int(16_000 * duration_s)
    pcm = np.zeros(n, dtype=np.float32)
    return lambda: voh.sources.from_numpy_array(pcm)


def test_pipeline_construction_wires_queues_subscribers_and_stages() -> None:
    """Construction sizes queues from config, disables the LLM, and registers subs.

    One construction scenario covering the wiring contract: every internal
    queue takes its capacity from the config's ``qsize_*`` fields; with no
    ``llm`` block no analyst stage is built; and each ``subscribe_*`` call
    registers exactly one callback on its stage.
    """
    cfg = voh.PipelineConfig(qsize_pcm=7, qsize_seg=3)  # llm=None
    p = voh.Pipeline(source=_silent_source(), config=cfg)

    # Queue sizing: PCM queue takes qsize_pcm, every segment queue qsize_seg.
    assert p._q_pcm.maxsize == 7
    assert p._q_voiced.maxsize == 3
    assert p._q_diar.maxsize == 3
    assert p._q_utt.maxsize == 3
    assert p._q_summary.maxsize == 3
    # No llm block → no analyst stage.
    assert p._llm is None

    seen: list[str] = []

    async def voiced_cb(_x: object) -> None:
        """No-op voiced-segment subscriber used only to occupy a slot."""
        seen.append("voiced")

    async def diar_cb(_x: object) -> None:
        """No-op diarized-segment subscriber used only to occupy a slot."""
        seen.append("diar")

    async def utt_cb(_x: object) -> None:
        """No-op utterance subscriber used only to occupy a slot."""
        seen.append("utt")

    # Each subscribe_* call appends exactly one callback to its stage list.
    p.subscribe_voiced(voiced_cb)
    p.subscribe_diarized(diar_cb)
    p.subscribe_utterances(utt_cb)
    assert len(p._voiced_subs) == 1
    assert len(p._diar_subs) == 1
    assert len(p._utt_subs) == 1


# ---------------------------------------------------------------------------
# OnlineDiarStage construction-time validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"join_threshold": -0.1}, "join_threshold"),  # below range
        ({"join_threshold": 2.5}, "join_threshold"),  # above range
        ({"ema_alpha": 0.0}, "ema_alpha"),  # at/below open lower bound
        ({"ema_alpha": 1.5}, "ema_alpha"),  # above range
    ],
)
def test_online_diar_rejects_out_of_range_params(kwargs: dict[str, float], match: str) -> None:
    """Out-of-range diar thresholds blow up at construction, not runtime.

    ``OnlineDiarStage`` validates ``join_threshold`` and ``ema_alpha`` in
    ``__init__`` so a mis-configured stage fails fast rather than deep in the
    live loop.

    Parameters
    ----------
    kwargs : dict of str to float
        Single out-of-range keyword to pass to the constructor.
    match : str
        Substring the raised ``ValueError`` message must contain.
    """
    from vocal_helper.diar import OnlineDiarStage

    with pytest.raises(ValueError, match=match):
        OnlineDiarStage(**kwargs)


# ---------------------------------------------------------------------------
# End-to-end event loop (model-free)
# ---------------------------------------------------------------------------


def test_pipeline_run_completes_with_empty_buffer() -> None:
    """Empty PCM buffer → ``None`` sentinel cascades, ``run`` ends without yielding.

    Drives a real :meth:`Pipeline.run` to exhaustion with the heavy VAD /
    diar / ASR stages swapped for no-op coroutines that only forward the
    ``None`` sentinel. This proves the sentinel propagates stage-to-stage and
    the loop shuts down cleanly on an empty source — without loading a model.
    """
    pcm = np.zeros(0, dtype=np.float32)

    p = voh.Pipeline(
        source=lambda: voh.sources.from_numpy_array(pcm),
        config=voh.PipelineConfig(),
    )

    async def passthrough_vad(inbox: asyncio.Queue, outbox: asyncio.Queue) -> None:
        """No-op stage : forward only the ``None`` sentinel, load no model."""
        while True:
            item = await inbox.get()
            if item is None:
                await outbox.put(None)
                return

    async def passthrough_diar(inbox: asyncio.Queue, outbox: asyncio.Queue) -> None:
        """Diar stub reusing the VAD passthrough (sentinel-forward only)."""
        await passthrough_vad(inbox, outbox)

    async def passthrough_asr(inbox: asyncio.Queue, outbox: asyncio.Queue) -> None:
        """ASR stub reusing the VAD passthrough (sentinel-forward only)."""
        await passthrough_vad(inbox, outbox)

    p._vad.run = passthrough_vad  # type: ignore[assignment]
    p._diar.run = passthrough_diar  # type: ignore[assignment]
    p._asr.run = passthrough_asr  # type: ignore[assignment]

    async def drive() -> list:
        """Run the pipeline to exhaustion and collect whatever it yields."""
        return [ev async for ev in p.run()]

    events = asyncio.run(drive())
    assert events == []
