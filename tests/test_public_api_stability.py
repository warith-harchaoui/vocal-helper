"""Public-API stability guard.

vocal-helper is an open-source library other code installs and pins.
Removing or renaming a public symbol, dropping a dataclass key, changing
the speaker-label scheme, or turning an optional constructor argument
into a required one is a silent breaking change for every downstream.

These tests freeze the *observable contract* so such a change fails CI
loudly instead of surfacing as a downstream crash after a version bump.
They only ever assert on vocal-helper's own surface — never on any
consumer. Additions (new symbols, new TypedDict keys, new keyword-only
arguments with defaults) are allowed; removals and incompatible
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

# Frozen exported symbols — every name must stay in ``__all__`` and resolve.
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

# Frozen TypedDict keys — each must remain a subset of the live keys.
_FROZEN_KEYS = {
    PcmFrame: {"t0", "sample_rate", "pcm"},
    VoicedSegment: {"t0", "t1", "sample_rate", "pcm"},
    DiarizedSegment: {"t0", "t1", "sample_rate", "speaker", "pcm"},
    Utterance: {"t0", "t1", "speaker", "text", "words", "language"},
    SummarySnapshot: {"t0", "summary", "recent", "model"},
}


def test_public_contract_stable() -> None:
    """Every frozen export, TypedDict key, and constructor signature stays intact.

    Consolidates the whole "importable + signature" side of the contract into a
    single sweep so any silent breaking change fails loudly:

    * every frozen name is still in ``__all__`` and resolves non-``None``;
    * no frozen TypedDict key was dropped (additions allowed);
    * both pipeline constructors keep ``source`` plus an optional ``config``;
    * both config dataclasses keep every documented field;
    * every stage stays constructible with zero positional args (new features
      must be keyword-only with defaults);
    * ``OfflineDiarStage`` keeps ``pyannote`` as its default backend.
    """
    # 1. Exported symbols — present in __all__ and importable.
    for name in _FROZEN_EXPORTS:
        assert name in voh.__all__, f"{name} dropped from vocal_helper.__all__"
        assert getattr(voh, name, None) is not None, f"vocal_helper.{name} is missing"

    # 2. TypedDict keys — frozen keys stay a subset of the live keys.
    for td, frozen in _FROZEN_KEYS.items():
        missing = frozen - set(td.__annotations__)
        assert not missing, f"{td.__name__} lost keys: {missing}"

    # 3. Pipeline constructors — ``source`` required, ``config`` optional.
    for cls in (voh.Pipeline, voh.OfflinePipeline):
        params = inspect.signature(cls).parameters
        assert "source" in params
        assert "config" in params and params["config"].default is not inspect.Parameter.empty

    # 4. Config dataclasses — every documented field survives.
    pc = voh.PipelineConfig()
    for f in ("vad", "eot", "diar", "asr", "llm", "qsize_pcm", "qsize_seg"):
        assert hasattr(pc, f), f"PipelineConfig lost field {f}"
    oc = voh.OfflinePipelineConfig()
    for f in ("diar", "asr", "llm", "qsize_pcm", "qsize_seg"):
        assert hasattr(oc, f), f"OfflinePipelineConfig lost field {f}"

    # 5. Stages — no non-variadic parameter may lack a default (old call sites
    #    that construct a stage with zero positional args must keep working).
    for cls in (voh.WhisperStage, voh.OnlineDiarStage, voh.OfflineDiarStage, voh.SileroVADStage):
        for name, p in inspect.signature(cls).parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            assert p.default is not inspect.Parameter.empty, (
                f"{cls.__name__}.{name} became a required argument"
            )

    # 6. Offline diarization default backend stays pyannote (nemo stays selectable).
    sig = inspect.signature(voh.OfflineDiarStage)
    assert sig.parameters["backend"].default == "pyannote"


def test_speaker_label_scheme_unchanged() -> None:
    """The ``S?`` sentinel and ``S<int>`` label scheme remain in diar.py source.

    A distinct contract from the import/signature sweep: downstreams parse the
    speaker labels *literally* from the emitted stream, so both the
    unknown-speaker sentinel and the ``S<int>`` id scheme are frozen at the
    source-text level.
    """
    src = Path(voh.__file__).with_name("diar.py").read_text()
    assert '"S?"' in src, "the 'S?' unknown-speaker sentinel was removed"
    assert re.search(r'f"S\{', src), "the 'S<int>' speaker-id scheme changed"
