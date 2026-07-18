"""
Unit tests for the torch-free sherpa-onnx offline diarization backend.

Module summary
--------------
Exercises the ``backend='sherpa'`` wiring added to
:class:`vocal_helper.diar.OfflineDiarStage` — the portable ONNX pipeline
(pyannote community-1 segmentation + NeMo TitaNet-large embedding + fast
clustering) selected by the 2026-07-18 diarization study (ADR 0002). No
models are loaded and no network is touched: these tests cover backend
selection, whole-buffer sizing, the model-path resolver's env / bundle
branches, and the clear error raised when nothing is configured. The
actual ONNX inference is left to the ``integration`` regression test,
which needs the real model files.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from vocal_helper.diar import (
    IDEAL_DURATION_S_SHERPA,
    OfflineDiarStage,
    _resolve_sherpa_models,
    _SherpaOfflineDiar,
)


def test_offline_stage_selects_whole_buffer_for_sherpa() -> None:
    """``backend='sherpa'`` runs whole-buffer (no chunking), like pyannote.

    sherpa clusters the entire recording inside one ``process`` call, so
    the stage's ``ideal_duration_s`` must default to the very-large sherpa
    constant — chunking would only cost DER.
    """
    stage = OfflineDiarStage(backend="sherpa")
    assert stage.ideal_duration_s == IDEAL_DURATION_S_SHERPA


def test_sherpa_offline_diar_exposes_backend_interface() -> None:
    """``_SherpaOfflineDiar`` matches the ``load`` + ``diarize`` backend contract.

    The offline stage calls ``load()`` then ``diarize(pcm, sr)`` and expects
    ``[(t0, t1, speaker), …]`` — the same shape as the pyannote / NeMo backends.
    """
    diar = _SherpaOfflineDiar()
    assert hasattr(diar, "load")
    params = list(inspect.signature(diar.diarize).parameters)
    assert params == ["pcm", "sr"]
    # Clustering knobs default to auto speaker count, sensible pruning.
    assert diar.num_clusters == -1
    assert diar.threshold == 0.5


def test_resolve_sherpa_models_prefers_env_override(tmp_path: Path, monkeypatch) -> None:
    """Explicit ``$VH_SHERPA_*`` paths win over any bundle lookup.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory for fake model files.
    monkeypatch : pytest.MonkeyPatch
        Sets the two model-path environment overrides.
    """
    seg = tmp_path / "seg.onnx"
    emb = tmp_path / "emb.onnx"
    seg.write_bytes(b"\x00")
    emb.write_bytes(b"\x00")
    monkeypatch.setenv("VH_SHERPA_SEGMENTATION", str(seg))
    monkeypatch.setenv("VH_SHERPA_EMBEDDING", str(emb))
    assert _resolve_sherpa_models() == (str(seg), str(emb))


def test_resolve_sherpa_models_reads_bundle(tmp_path: Path, monkeypatch) -> None:
    """Without env overrides, the resolver finds ONNX files in the bundle's ``sherpa/``.

    Prefers our sovereign community-1 export for segmentation and TitaNet-large
    for the embedding — the study-selected offline pair.

    Parameters
    ----------
    tmp_path : Path
        Fake diarization-engines bundle root.
    monkeypatch : pytest.MonkeyPatch
        Clears the env overrides and points ``$VH_DIARIZATION_ENGINES`` at the bundle.
    """
    monkeypatch.delenv("VH_SHERPA_SEGMENTATION", raising=False)
    monkeypatch.delenv("VH_SHERPA_EMBEDDING", raising=False)
    (tmp_path / "manifest.json").write_text("{}")
    sdir = tmp_path / "sherpa"
    sdir.mkdir()
    (sdir / "community1-segmentation.onnx").write_bytes(b"\x00")
    (sdir / "nemo_en_titanet_large.onnx").write_bytes(b"\x00")
    monkeypatch.setenv("VH_DIARIZATION_ENGINES", str(tmp_path))

    seg, emb = _resolve_sherpa_models()
    assert seg == str(sdir / "community1-segmentation.onnx")
    assert emb == str(sdir / "nemo_en_titanet_large.onnx")


def test_resolve_sherpa_models_raises_when_unconfigured(monkeypatch) -> None:
    """A clear ``RuntimeError`` is raised when no models can be found.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Clears the env overrides and points the bundle resolver at a
        non-existent path so it resolves to ``None`` (no download attempted).
    """
    monkeypatch.delenv("VH_SHERPA_SEGMENTATION", raising=False)
    monkeypatch.delenv("VH_SHERPA_EMBEDDING", raising=False)
    monkeypatch.setenv("VH_DIARIZATION_ENGINES", "/nope/does/not/exist")
    with pytest.raises(RuntimeError, match="segmentation ONNX"):
        _resolve_sherpa_models()
