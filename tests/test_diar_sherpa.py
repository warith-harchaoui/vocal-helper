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


def test_sherpa_stage_wiring_and_backend_contract() -> None:
    """``backend='sherpa'`` runs whole-buffer and exposes the load+diarize contract.

    sherpa clusters the entire recording inside one ``process`` call, so the
    stage's ``ideal_duration_s`` must default to the very-large sherpa constant
    (chunking would only cost DER). The backend itself must match the same
    ``load()`` then ``diarize(pcm, sr) -> [(t0, t1, speaker), …]`` shape as the
    pyannote / NeMo backends, defaulting to auto speaker count and sensible
    pruning.
    """
    # Whole-buffer sizing: no chunking for the clustering backend.
    stage = OfflineDiarStage(backend="sherpa")
    assert stage.ideal_duration_s == IDEAL_DURATION_S_SHERPA
    # Backend contract: load + diarize(pcm, sr), auto-cluster defaults.
    diar = _SherpaOfflineDiar()
    assert hasattr(diar, "load")
    assert list(inspect.signature(diar.diarize).parameters) == ["pcm", "sr"]
    assert diar.num_clusters == -1
    assert diar.threshold == 0.5


def test_resolve_sherpa_models_env_override_wins_then_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit ``$VH_SHERPA_*`` paths win; without them the bundle's ``sherpa/`` is read.

    Two resolution success paths in one scenario. First, both env overrides set:
    the resolver returns them verbatim and never consults the bundle (the
    operator path wins). Then, with the overrides cleared, it discovers the
    study-selected community-1 segmentation + TitaNet-large embedding ONNX files
    by name inside the bundle's ``sherpa/`` directory.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory holding the fake override files and the fake bundle.
    monkeypatch : pytest.MonkeyPatch
        Sets / clears the model-path env overrides and the bundle root.
    """
    # Env-override branch: two non-empty stand-in ONNX files (the resolver only
    # checks existence, never parses, so a single null byte reads as "present").
    seg_env = tmp_path / "seg.onnx"
    emb_env = tmp_path / "emb.onnx"
    seg_env.write_bytes(b"\x00")
    emb_env.write_bytes(b"\x00")
    monkeypatch.setenv("VH_SHERPA_SEGMENTATION", str(seg_env))
    monkeypatch.setenv("VH_SHERPA_EMBEDDING", str(emb_env))
    # Explicit overrides returned verbatim, bundle never consulted.
    assert _resolve_sherpa_models() == (str(seg_env), str(emb_env))

    # Bundle branch: clear the overrides so the resolver falls to the bundle.
    monkeypatch.delenv("VH_SHERPA_SEGMENTATION", raising=False)
    monkeypatch.delenv("VH_SHERPA_EMBEDDING", raising=False)
    # A minimal bundle: a manifest marks the root, and sherpa/ holds the two
    # study-selected ONNX exports the resolver is expected to discover by name.
    bundle = tmp_path / "bundle"
    (bundle / "sherpa").mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}")
    sdir = bundle / "sherpa"
    (sdir / "community1-segmentation.onnx").write_bytes(b"\x00")
    (sdir / "nemo_en_titanet_large.onnx").write_bytes(b"\x00")
    monkeypatch.setenv("VH_DIARIZATION_ENGINES", str(bundle))

    seg, emb = _resolve_sherpa_models()
    assert seg == str(sdir / "community1-segmentation.onnx")
    assert emb == str(sdir / "nemo_en_titanet_large.onnx")


def test_resolve_sherpa_models_raises_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
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
