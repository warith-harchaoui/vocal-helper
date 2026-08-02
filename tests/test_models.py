"""
Unit tests for the shared-mirror model registry (:mod:`vocal_helper.models`).

Module summary
--------------
Covers the registry + on-demand downloader that backs the torch-free ``sherpa``
diarization path: the registry lists the ONNX weights the path needs, and
``ensure_model`` resolves each to a cached local path from the shared ai-helpers
mirror. No network is touched — the download primitive (``os_helper.download_file``)
is stubbed — so the tests assert the *wiring*: cache reuse, the mirror URL built
from the (overridable) base, the license gate, and graceful failure.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vocal_helper import models


def test_registry_covers_the_sherpa_path_models() -> None:
    """The registry lists exactly the ONNX weights the sherpa path resolves.

    Each entry's filename must match the bare name ``diar._resolve_sherpa_models``
    looks for in the bundle, so the mirror is a drop-in resolution tier.
    """
    expected = {
        "sherpa-titanet-large": "nemo_en_titanet_large.onnx",
        "sherpa-titanet-small": "nemo_en_titanet_small.onnx",
        "community1-segmentation": "community1-segmentation.onnx",
        "pyannote-segmentation-3": "pyannote_segmentation_3.onnx",
    }
    assert {k: v.filename for k, v in models.REGISTRY.items()} == expected
    # Every listed weight is under a commercial-OK license → not gated by default.
    for spec in models.REGISTRY.values():
        assert spec.license in models._PERMISSIVE_LICENSES


def test_ensure_model_reuses_cache_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An already-cached weight is returned verbatim, no download attempted.

    Parameters
    ----------
    tmp_path : Path
        Stands in for the shared model cache via ``VOCAL_HELPER_MODEL_DIR``.
    monkeypatch : pytest.MonkeyPatch
        Redirects the cache dir and makes ``download_file`` fail loudly if called.
    """
    monkeypatch.setenv("VOCAL_HELPER_MODEL_DIR", str(tmp_path))
    (tmp_path / "nemo_en_titanet_large.onnx").write_bytes(b"\x00")

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("download_file must not be called when the cache is warm")

    monkeypatch.setattr(models.osh, "download_file", _boom)
    path = models.ensure_model("sherpa-titanet-large")
    assert path == str(tmp_path / "nemo_en_titanet_large.onnx")


def test_ensure_model_downloads_from_overridable_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a cold cache the weight is fetched from ``AI_HELPERS_MODEL_BASE_URL``.

    Parameters
    ----------
    tmp_path : Path
        The cold cache directory.
    monkeypatch : pytest.MonkeyPatch
        Redirects the cache dir, overrides the mirror base, and records the URL
        the stubbed ``download_file`` is asked to fetch.
    """
    monkeypatch.setenv("VOCAL_HELPER_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("AI_HELPERS_MODEL_BASE_URL", "https://example.test/models/")
    seen: dict[str, str] = {}

    def _fake_download(url: str, dest: str = "") -> None:
        seen["url"] = url
        Path(dest).write_bytes(b"\x00")

    monkeypatch.setattr(models.osh, "download_file", _fake_download)
    path = models.ensure_model("community1-segmentation")
    assert path == str(tmp_path / "community1-segmentation.onnx")
    assert seen["url"] == "https://example.test/models/community1-segmentation.onnx"


def test_ensure_model_gates_noncommercial_and_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown keys and non-permissive licenses return ``None`` (caller degrades).

    Parameters
    ----------
    tmp_path : Path
        Cache dir (unused on the gated paths, but isolates the test).
    monkeypatch : pytest.MonkeyPatch
        Redirects the cache dir and injects a research-licensed spec.
    """
    monkeypatch.setenv("VOCAL_HELPER_MODEL_DIR", str(tmp_path))
    # A download at any point here would be a bug (both calls should short-circuit
    # on the gate, and the allow path is served by the stub, not the network).
    monkeypatch.setattr(
        models.osh, "download_file", lambda url, dest="": Path(dest).write_bytes(b"\x00")
    )
    assert models.ensure_model("does-not-exist") is None

    gated = models.ModelSpec(name="gated", filename="gated.onnx", license="research")
    monkeypatch.setitem(models.REGISTRY, "gated", gated)
    # Gated off by default; the escape hatch lets it through to the (stubbed) fetch.
    assert models.ensure_model("gated") is None
    assert models.ensure_model("gated", allow_noncommercial=True) == str(tmp_path / "gated.onnx")
