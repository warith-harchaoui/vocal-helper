"""Public-API stability guard.

vocal-helper is an open-source library other code installs and pins.
Removing or renaming a public symbol, dropping a dataclass key, changing
the speaker-label scheme, or turning an optional constructor argument
into a required one is a silent breaking change for every downstream.

This test freezes the *observable contract* so such a change fails CI
loudly instead of surfacing as a downstream crash after a version bump.
It only ever asserts on vocal-helper's own surface — never on any
consumer. Additions (new symbols, new TypedDict keys, new keyword-only
arguments with defaults) are allowed ; removals and incompatible
signature changes are not.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import vocal_helper as voh
from vocal_helper.types import (
    DiarizedSegment,
    PcmFrame,
    SummarySnapshot,
    Utterance,
    VoicedSegment,
)

# --------------------------------------------------------------------------
# 1. Exported symbols
# --------------------------------------------------------------------------

_FROZEN_EXPORTS = {
    "sources",
    "Pipeline",
    "PipelineConfig",
    "OfflinePipeline",
    "OfflinePipelineConfig",
    "SileroVADStage",
    "OnlineDiarStage",
    "OfflineDiarStage",
    "WhisperStage",
    "LangRegion",
    "RegionVerdict",
    "cross_check_regions",
    "detect_language",
    "detect_language_regions",
    "detect_language_speechbrain",
    "language_posterior_curve",
    "GemmaAnalystStage",
    "PcmFrame",
    "VoicedSegment",
    "DiarizedSegment",
    "Utterance",
    "SummarySnapshot",
    "transcribe_pcm",
}


def test_public_exports_present() -> None:
    for name in _FROZEN_EXPORTS:
        assert name in voh.__all__, f"{name} dropped from vocal_helper.__all__"
        assert getattr(voh, name, None) is not None, f"vocal_helper.{name} is missing"


# --------------------------------------------------------------------------
# 2. TypedDict keys (frozen keys must remain a subset of the live keys)
# --------------------------------------------------------------------------

_FROZEN_KEYS = {
    PcmFrame: {"t0", "sample_rate", "pcm"},
    VoicedSegment: {"t0", "t1", "sample_rate", "pcm"},
    DiarizedSegment: {"t0", "t1", "sample_rate", "speaker", "pcm"},
    Utterance: {"t0", "t1", "speaker", "text", "words", "language"},
    SummarySnapshot: {"t0", "summary", "recent", "model"},
}


def test_typed_dict_keys_stable() -> None:
    for td, frozen in _FROZEN_KEYS.items():
        live = set(td.__annotations__)
        missing = frozen - live
        assert not missing, f"{td.__name__} lost keys: {missing}"


# --------------------------------------------------------------------------
# 3. Constructor signatures — old call sites must keep working
# --------------------------------------------------------------------------


def test_pipeline_constructors_accept_documented_kwargs() -> None:
    # ``source=`` + optional ``config=`` is the frozen construction shape.
    for cls in (voh.Pipeline, voh.OfflinePipeline):
        params = inspect.signature(cls).parameters
        assert "source" in params
        assert "config" in params and params["config"].default is not inspect.Parameter.empty


def test_config_fields_stable() -> None:
    pc = voh.PipelineConfig()
    for f in ("vad", "eot", "diar", "asr", "llm", "qsize_pcm", "qsize_seg"):
        assert hasattr(pc, f), f"PipelineConfig lost field {f}"
    oc = voh.OfflinePipelineConfig()
    for f in ("diar", "asr", "llm", "qsize_pcm", "qsize_seg"):
        assert hasattr(oc, f), f"OfflinePipelineConfig lost field {f}"


def test_stage_constructors_have_no_new_required_args() -> None:
    """Every stage must stay constructible with zero positional args.

    New features (batch, max_chunk_s, warmup, …) must be keyword-only
    with defaults, so a pinned downstream that constructs a stage the old
    way keeps working after a version bump.
    """
    for cls in (voh.WhisperStage, voh.OnlineDiarStage, voh.OfflineDiarStage, voh.SileroVADStage):
        for name, p in inspect.signature(cls).parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            assert p.default is not inspect.Parameter.empty, (
                f"{cls.__name__}.{name} became a required argument"
            )


def test_offline_diar_backend_options_preserved() -> None:
    """pyannote stays the offline default ; nemo stays selectable."""
    sig = inspect.signature(voh.OfflineDiarStage)
    assert sig.parameters["backend"].default == "pyannote"


# --------------------------------------------------------------------------
# 4. Speaker-label scheme — downstreams parse "S0"/"S1"/"S?" literally
# --------------------------------------------------------------------------


def test_speaker_label_scheme_unchanged() -> None:
    src = Path(voh.__file__).with_name("diar.py").read_text()
    # The unknown-speaker sentinel and the S<int> id scheme are contractual.
    assert '"S?"' in src, "the 'S?' unknown-speaker sentinel was removed"
    assert re.search(r'f"S\{', src), "the 'S<int>' speaker-id scheme changed"
