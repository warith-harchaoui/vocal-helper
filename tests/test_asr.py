"""WhisperStage — construction, batched offline path, and one-shot helpers.

All tests are model-free: the whisper.cpp model is stubbed or replaced so no
real backend is loaded. They cover the construction contract (defaults +
overrides + pipeline wiring), the batched decode's "one utterance per segment"
invariant (success and failure), the packing/windowing helpers that feed it,
and the discovery-first one-shot helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from vocal_helper.asr import (
    DEFAULT_INITIAL_PROMPT,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHUNK_S,
    DEFAULT_MIN_SEGMENT_MS,
    DEFAULT_MODEL,
    WhisperStage,
    _assign_window,
    _pack_segments,
    transcribe_pcm,
    transcribe_pcm_with_language,
)
from vocal_helper.types import DiarizedSegment, Utterance

SR = 16000


def _seg(t0: float, t1: float, speaker: str) -> DiarizedSegment:
    """Build a silent :class:`DiarizedSegment` spanning ``[t0, t1]`` for one speaker."""
    n = int(round((t1 - t0) * SR))
    return DiarizedSegment(
        t0=t0,
        t1=t1,
        sample_rate=SR,
        speaker=speaker,
        pcm=np.zeros(n, dtype=np.float32),
    )


# ----- construction contract ------------------------------------------------


def test_whisper_construction_contract() -> None:
    """Defaults are canonical and lazy; every constructor override is stored verbatim.

    Merges the defaults, override, empty-prompt, and batch-off checks into one
    construction contract: a bare stage carries the documented defaults (and
    hasn't loaded a model), while an explicitly-configured stage echoes each
    field back unchanged.
    """
    stage = WhisperStage()
    assert stage.model_name == DEFAULT_MODEL
    assert stage.language == DEFAULT_LANGUAGE
    # The thread count is no longer a constant. It is sized to THIS machine's
    # performance cores, because a hard-coded number either leaves fast cores idle
    # or spills onto efficiency cores — and spilling makes the stage slower, since
    # every synchronisation point then waits on the slowest participant. What the
    # contract pins is the SOURCE of the number, not the number.
    from vocal_helper.accel import probe

    assert stage.threads == probe().performance_threads >= 1
    # And the stage records which accelerator it resolved, so a surprising
    # real-time factor can be explained rather than guessed at.
    assert stage.accel_plan.accelerator in ("ane", "coreml", "metal", "cpu")
    assert stage.accel_plan.why
    assert stage.word_timestamps is True
    # Empty default keeps the zero-config path working; the docstring still
    # recommends a domain-aligned prompt.
    assert stage.initial_prompt == DEFAULT_INITIAL_PROMPT == ""
    assert stage.min_segment_ms == DEFAULT_MIN_SEGMENT_MS
    assert stage._model is None  # lazy — model not loaded until first use
    # Batching + warm-up are opt-in; a bare stage keeps per-segment behaviour.
    assert stage.batch is False
    assert stage.max_chunk_s == DEFAULT_MAX_CHUNK_S
    assert stage.warmup is False

    # Every override is stored as given.
    stage = WhisperStage(
        model="base",
        language="fr",
        threads=4,
        word_timestamps=False,
        initial_prompt="meeting transcript: scope, budget, deliverables",
        min_segment_ms=500,
    )
    assert stage.model_name == "base"
    assert stage.language == "fr"
    assert stage.threads == 4
    assert stage.word_timestamps is False
    assert stage.initial_prompt.startswith("meeting transcript")
    assert stage.min_segment_ms == 500


def test_pipeline_wiring_defaults() -> None:
    """Offline defaults to batched ASR; streaming defaults to warm-up; config wins."""
    import vocal_helper as voh

    src = lambda: iter(())  # noqa: E731 — never run, just to construct.
    online = voh.Pipeline(source=src)
    assert online._asr.batch is False
    assert online._asr.warmup is True

    # Offline with the language to discover, the measured best engine is
    # Nemotron 3.5; naming a language keeps whisper (``choose_asr_engine``).
    offline = voh.OfflinePipeline(source=src)
    assert type(offline._asr).__name__ == "NemotronAsrStage"
    assert offline._asr.batch is True
    offline = voh.OfflinePipeline(
        source=src, config=voh.OfflinePipelineConfig(asr={"language": "fr"})
    )
    assert offline._asr.batch is True
    assert offline._asr.warmup is False

    # Explicit config always wins over the pipeline default.
    off2 = voh.OfflinePipeline(source=src, config=voh.OfflinePipelineConfig(asr={"batch": False}))
    assert off2._asr.batch is False


# ----- packing + windowing helpers ------------------------------------------


def test_pack_and_window_helpers() -> None:
    """``_pack_segments`` respects the chunk cap and ``_assign_window`` maps times.

    Two helpers that feed the batched decode, checked together: packing groups
    segments up to ``max_chunk_s`` (isolating over-cap ones, collapsing under a
    huge cap, splitting under a tiny one), and window assignment resolves a time
    to its containing window or the nearest edge otherwise.
    """
    segs = [_seg(0, 2, "S0"), _seg(2, 4, "S1"), _seg(4, 6, "S0"), _seg(6, 20, "S1")]
    # cap 5 s : [2+2] together (4 + pad ≤ 5), then the 2 s, then the 14 s alone.
    assert [len(c) for c in _pack_segments(segs, max_chunk_s=5.0)] == [2, 1, 1]
    # cap huge : everything in one chunk.
    assert len(_pack_segments(segs, max_chunk_s=1000.0)) == 1
    # cap tiny : every segment isolated.
    assert [len(c) for c in _pack_segments(segs, max_chunk_s=0.5)] == [1, 1, 1, 1]

    windows = [(0.0, 2.0), (2.1, 4.1), (4.2, 6.2)]
    assert _assign_window(windows, 1.0) == 0
    assert _assign_window(windows, 3.0) == 1
    assert _assign_window(windows, 6.0) == 2
    assert _assign_window(windows, 2.05) in (0, 1)  # in a pad gap → nearest edge
    assert _assign_window(windows, 99.0) == 2  # past the end → last window


# ----- batched offline decode -----------------------------------------------


class _Phrase:
    """Minimal stand-in for a whisper.cpp phrase (text + centisecond timings)."""

    def __init__(self, text: str, t0_cs: float, t1_cs: float, language: str | None = None):
        """Store the phrase text, chunk-local timings, and optional language tag."""
        self.text = text
        self.t0 = t0_cs  # centiseconds, chunk-local (whisper.cpp convention)
        self.t1 = t1_cs
        self.language = language


class _StubModel:
    """Returns phrases positioned to fall inside specific segment windows."""

    def __init__(self, phrases: list[_Phrase]):
        """Capture the canned phrases to replay on every ``transcribe`` call."""
        self._phrases = phrases

    def transcribe(self, pcm, **kwargs):  # noqa: ANN001
        """Ignore the audio and return the pre-positioned phrases."""
        return self._phrases


def test_batched_chunk_one_utterance_per_segment() -> None:
    """Batched decode emits exactly one utterance per segment — on success and on crash.

    The invariant is the same in both paths: N input segments → N utterances,
    in order, each tagged with its diarization speaker. On success, phrases are
    routed to their owning segment and word times are mapped back to the real
    timeline; on a model crash, each segment still yields an empty utterance
    (never fewer).
    """
    # Success: three segments S0 [0,2], S1 [2,3], S0 [3,5]. Concatenated with
    # 0.1 s pads, local windows ≈ [0,2] [2.1,3.1] [3.2,5.2].
    chunk = [_seg(0, 2, "S0"), _seg(2, 3, "S1"), _seg(3, 5, "S0")]
    stage = WhisperStage(batch=True)
    stage._model = _StubModel(
        [
            _Phrase("hello", 50, 150, "en"),  # mid 1.0 → window 0 (S0)
            _Phrase("bonjour", 250, 300, "fr"),  # mid 2.75 → window 1 (S1)
            _Phrase("world", 340, 500, "en"),  # mid 4.2 → window 2 (S0)
        ]
    )
    utts = stage._transcribe_chunk_blocking(chunk)
    # Exactly one Utterance per input segment, in order, carrying the diar speaker.
    assert len(utts) == 3
    assert [u["speaker"] for u in utts] == ["S0", "S1", "S0"]
    assert [u["t0"] for u in utts] == [0, 2, 3]
    assert [u["text"] for u in utts] == ["hello", "bonjour", "world"]
    assert utts[1]["language"] == "fr"
    # Word abs-time is mapped back into the owning segment's real timeline.
    w_t0, _w_t1, w_text = utts[1]["words"][0]
    assert w_text == "bonjour"
    assert 2.0 <= w_t0 <= 3.0  # inside S1's [2,3]

    # Failure: a model that always raises still preserves one empty utterance
    # per segment, so downstream indexing never desyncs.
    class _Boom:
        """A model whose ``transcribe`` always raises, to test the failure path."""

        def transcribe(self, pcm, **kwargs):  # noqa: ANN001
            """Simulate a whisper.cpp blow-up mid-decode."""
            raise RuntimeError("whisper exploded")

    stage = WhisperStage(batch=True)
    stage._model = _Boom()
    utts = stage._transcribe_chunk_blocking([_seg(0, 2, "S0"), _seg(2, 4, "S1")])
    assert [u["speaker"] for u in utts] == ["S0", "S1"]
    assert all(u["text"] == "" for u in utts)


# ----- transcribe_pcm / _with_language (discovery-first, model-free) ---------


def _stub_whisper(monkeypatch: pytest.MonkeyPatch, text: str, language: str | None) -> None:
    """Patch :class:`WhisperStage` so no real whisper.cpp model is loaded.

    ``_ensure_model`` becomes a no-op and ``_transcribe_blocking`` returns a
    canned :class:`Utterance` carrying ``text`` and the *discovered* ``language``
    — exactly what the one-shot helpers read back.
    """
    # No model load: the helpers call _ensure_model() before transcribing.
    monkeypatch.setattr(WhisperStage, "_ensure_model", lambda self: None)

    def _fake_transcribe(self: WhisperStage, seg: DiarizedSegment) -> Utterance:
        """Return a fixed utterance, echoing the segment's timing/speaker."""
        return Utterance(
            t0=seg["t0"],
            t1=seg["t1"],
            speaker=seg["speaker"],
            text=text,
            words=[],
            language=language,
        )

    monkeypatch.setattr(WhisperStage, "_transcribe_blocking", _fake_transcribe)


def test_one_shot_helpers_surface_discovered_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helpers return whisper's *discovered* language, and ``transcribe_pcm``
    keeps its text-only contract.

    ``transcribe_pcm_with_language`` surfaces the language whisper actually found
    (not the "auto" the caller passed), while ``transcribe_pcm`` delegates to it
    and hands back only the text string.
    """
    # whisper "discovers" French even though the caller left language on "auto".
    _stub_whisper(monkeypatch, "bonjour", "fr")
    text, language = transcribe_pcm_with_language(np.zeros(SR, dtype=np.float32), sr=SR)
    assert text == "bonjour"
    assert language == "fr"

    # transcribe_pcm delegates to the language helper but returns only the text.
    _stub_whisper(monkeypatch, "hola", "es")
    out = transcribe_pcm(np.zeros(SR, dtype=np.float32), sr=SR)
    assert out == "hola"
    assert isinstance(out, str)


# --------------------------------------------------------------------------- #
#  Acceleration: the hardware decides, and it says so
# --------------------------------------------------------------------------- #


def test_the_accelerator_is_resolved_from_the_hardware_not_assumed() -> None:
    """A wheel built without Metal looks identical from the outside and costs a
    measured 4.9× on the same model and audio. So the backend is PROBED, and the
    probe's conclusion travels with the stage."""
    from vocal_helper.accel import probe, resolve

    hardware = probe()
    plan = resolve("auto")
    if hardware.whisper_backends & {"ane", "coreml", "metal"}:
        assert plan.accelerator in hardware.whisper_backends
    else:
        assert plan.accelerator == "cpu"
        assert "five times" in plan.why  # the cost of the fallback is stated, not hidden


def test_an_explicit_accelerator_is_honoured_even_when_absent() -> None:
    """An override exists for the case where the probe is wrong, so it must not be
    overruled by the probe. But the plan records the discrepancy: a silent override
    plus a disappointing result is an unchaseable bug."""
    from vocal_helper.accel import probe, resolve

    plan = resolve("ane")
    assert plan.accelerator == "ane"
    if "ane" not in probe().whisper_backends:
        assert "not in the installed build" in plan.why


def test_asking_for_the_cpu_actually_asks_for_it() -> None:
    """whisper.cpp defaults to the GPU, so the CPU path is not the absence of a
    choice — it has to be requested, or a 'CPU' benchmark silently measures Metal."""
    from vocal_helper.accel import resolve

    assert resolve("cpu").model_kwargs == {"context_params": {"use_gpu": False}}


def test_an_unknown_accelerator_fails_at_the_decision_not_in_a_constructor() -> None:
    from vocal_helper.accel import resolve

    with pytest.raises(ValueError, match="unknown accelerator"):
        resolve("tensorrt")


def test_stage_real_time_factors_are_not_additive() -> None:
    """The arithmetic that matters live. Stages on DIFFERENT units overlap in the
    producer/consumer pipeline, so the chain costs the slower one; stages on the
    SAME unit queue for it, and there the costs add.

    Summing everything, which is the obvious reading, overstates a well-placed
    chain and hides the real problem — never the total, always one saturated unit.
    """
    from vocal_helper.accel import StageLoad, live_budget

    spread = live_budget([StageLoad("asr", "gpu", 0.40), StageLoad("diar", "cpu", 0.40)])
    assert spread.effective_rtf == 0.40  # overlapped, not 0.80
    assert spread.holds()

    piled = live_budget([StageLoad("asr", "cpu", 0.40), StageLoad("diar", "cpu", 0.40)])
    assert piled.effective_rtf == 0.80  # serialised on one unit
    assert not piled.holds()
    assert piled.bottleneck == "cpu"


def test_the_default_live_placement_holds_with_room_to_spare() -> None:
    """The measured default — whisper.cpp on Metal plus the NeMo online embedder,
    both on the GPU, with Silero on the CPU. They share the GPU and therefore
    serialise, and it still leaves roughly sixteen times real time."""
    from vocal_helper.accel import MEASURED_RTF, StageLoad, live_budget

    budget = live_budget(
        [
            StageLoad("vad", "cpu", MEASURED_RTF[("vad_silero", "cpu")]),
            StageLoad("asr", "gpu", MEASURED_RTF[("asr", "metal")]),
            StageLoad("diar", "gpu", MEASURED_RTF[("diar_online_nemo", "gpu")]),
        ]
    )
    assert budget.bottleneck == "gpu"
    assert budget.headroom > 10
    assert budget.holds(margin=5.0)


def test_the_torch_free_path_on_a_machine_without_a_gpu_is_the_tight_one() -> None:
    """Everything lands on the CPU and the margin collapses to about 1.3×. One
    slower machine, one bigger model or one competing process crosses 1 — and
    crossing 1 is not a graceful degradation, it is a delay that grows for the rest
    of the session. This is the case `source_policy="drop_oldest"` exists to bound.
    """
    from vocal_helper.accel import MEASURED_RTF, StageLoad, live_budget

    budget = live_budget(
        [
            StageLoad("vad", "cpu", MEASURED_RTF[("vad_silero", "cpu")]),
            StageLoad("asr", "cpu", MEASURED_RTF[("asr", "cpu")]),
            StageLoad("diar", "cpu", MEASURED_RTF[("diar_online_sherpa", "cpu")]),
        ]
    )
    assert budget.bottleneck == "cpu"
    assert 1.2 < budget.headroom < 1.5
    assert not budget.holds()


def test_the_default_model_is_turbo_large_v3() -> None:
    """The model family is a measured decision, not a preference, and it is the one
    every real-time factor in this repository was measured against.

    ``large-v3-turbo`` is the whole reason the numbers hold: it is a four-layer
    decoder against large-v3's thirty-two, which is where the speed comes from at a
    word-error rate close enough to be worth it. Dropping to ``base`` would make the
    benchmarks meaningless; climbing to full ``large-v3`` would cost roughly an
    order of magnitude and sink the live budget.

    The quantisation suffix is deliberately NOT pinned: it is a tuning knob, and a
    future q8 or a non-quantised build is a legitimate trade. The family is not.

    **And it is already the newest one.** Checked on 2026-11-01 against OpenAI's own
    published model list: the Whisper line stops at ``large-v3-turbo`` (October 2024,
    0.8 billion parameters). There is no ``large-v4`` and no ``large-v4-turbo``.
    OpenAI's later transcription models are API-only and closed, so they are not
    available to a local-first free path at all — which is the path this stage is.
    """
    from vocal_helper.asr import DEFAULT_MODEL

    assert DEFAULT_MODEL.startswith("large-v3-turbo"), (
        f"default model is {DEFAULT_MODEL!r}; every RTF in studies/ was measured on "
        f"large-v3-turbo, so changing the family invalidates them"
    )
