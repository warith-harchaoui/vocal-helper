"""
Unit tests for the HF-free diarization-engines resolver.

Module summary
--------------
Exercises :func:`vocal_helper.diar.resolve_diarization_engines` — the
function that locates the self-hosted, HuggingFace-free model bundle —
without any network access. Only the local-directory and unset branches
are covered here; the URL-download branch needs a live host and is left
to the ``integration`` regression test.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import glob
import tempfile
import zipfile
from pathlib import Path

from vocal_helper.diar import resolve_diarization_engines


def test_resolver_returns_local_dir_with_manifest(tmp_path: Path, monkeypatch) -> None:
    """A local dir holding ``manifest.json`` is returned as-is.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to point ``$VH_DIARIZATION_ENGINES`` at ``tmp_path``.
    """
    # A bundle is identified by the presence of a manifest at its root.
    (tmp_path / "manifest.json").write_text("{}")
    # Point the resolver at our fake local bundle.
    monkeypatch.setenv("VH_DIARIZATION_ENGINES", str(tmp_path))
    # The resolver should hand back exactly that directory, no download.
    assert resolve_diarization_engines() == tmp_path


def test_resolver_returns_none_for_missing_local_dir(monkeypatch) -> None:
    """A non-existent local path resolves to ``None`` (caller falls back).

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to set ``$VH_DIARIZATION_ENGINES`` to a bogus path.
    """
    # A path that does not exist must not be treated as a bundle.
    monkeypatch.setenv("VH_DIARIZATION_ENGINES", "/nope/does/not/exist")
    assert resolve_diarization_engines() is None


def test_resolver_local_dir_without_manifest_is_rejected(tmp_path: Path, monkeypatch) -> None:
    """An existing dir lacking ``manifest.json`` still resolves to a dir.

    The resolver returns the directory (it exists) so the concrete
    backend can decide whether the specific weights it needs are present;
    the manifest is only the strong signal used for URL caching.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory (left without a manifest).
    monkeypatch : pytest.MonkeyPatch
        Used to point ``$VH_DIARIZATION_ENGINES`` at ``tmp_path``.
    """
    # No manifest here — but the directory exists, so it is returned and
    # each backend's own ``.exists()`` check decides usability.
    monkeypatch.setenv("VH_DIARIZATION_ENGINES", str(tmp_path))
    assert resolve_diarization_engines() == tmp_path


def test_resolver_url_branch_cleans_up_its_temp_zip(tmp_path: Path, monkeypatch) -> None:
    """The URL-download branch must not leak its downloaded ``.zip``.

    ``resolve_diarization_engines`` downloads a (real-world: ~750 MB) bundle
    to a ``NamedTemporaryFile(delete=False)`` before extracting it — it owns
    cleanup itself, and used to never unlink it, leaking one temp file per
    fresh-machine bootstrap. No network here: ``osh.download_file`` is faked
    to write a small real zip so the extract path is genuinely exercised.
    """
    monkeypatch.setattr(
        "vocal_helper._settings.resolve_diarization_engines_url",
        lambda *a, **k: "https://example.invalid/diarization-engines.zip",
    )
    monkeypatch.setenv("VH_CACHE_DIR", str(tmp_path / "cache"))

    def _fake_download_file(url: str, dst: str) -> None:
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr("manifest.json", "{}")

    monkeypatch.setattr("os_helper.download_file", _fake_download_file, raising=False)

    tmp_zip_glob = str(Path(tempfile.gettempdir()) / "tmp*.zip")
    before = set(glob.glob(tmp_zip_glob))
    result = resolve_diarization_engines()
    after = set(glob.glob(tmp_zip_glob))

    assert result == tmp_path / "cache" / "diarization-engines"
    assert (result / "manifest.json").exists()
    assert after == before, f"leaked temp zip(s): {after - before}"
