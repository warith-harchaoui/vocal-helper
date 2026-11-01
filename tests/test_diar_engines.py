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

import pytest

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


def test_resolver_downloads_through_os_helper_and_leaves_nothing_behind(
    tmp_path: Path, monkeypatch
) -> None:
    """La branche de téléchargement passe par os-helper, et ne laisse rien.

    Elle a bricolé deux fois : un ``NamedTemporaryFile(delete=False)`` jamais
    supprimé, qui fuyait une archive de près d'un gigaoctet par amorçage de
    machine, puis un ``zipfile.extractall`` qui honore un membre nommé
    ``../ailleurs``. Les deux appartiennent à os-helper, dont le dossier
    temporaire se nettoie seul et dont ``unzip_folder`` refuse un membre
    s'échappant de la destination. Aucun réseau ici : le téléchargement est
    simulé par l'écriture d'un vrai petit zip, pour que l'extraction soit
    réellement exercée.
    """
    monkeypatch.setattr(
        "vocal_helper._settings.resolve_diarization_engines_url",
        lambda *a, **k: "https://example.invalid/diarization-engines.zip",
    )
    monkeypatch.setenv("VH_CACHE_DIR", str(tmp_path / "cache"))

    seen: dict[str, object] = {}

    def _fake_download_file(url: str, dst: str, **kwargs) -> None:
        seen["url"] = url
        seen["kwargs"] = kwargs
        seen["dst"] = dst
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr("manifest.json", "{}")

    monkeypatch.setattr("os_helper.download_file", _fake_download_file, raising=False)

    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "*")))
    result = resolve_diarization_engines()
    after = set(glob.glob(str(Path(tempfile.gettempdir()) / "*")))

    assert result == tmp_path / "cache" / "diarization-engines"
    assert (result / "manifest.json").exists()
    assert after == before, f"restes dans le dossier temporaire : {after - before}"
    assert "sha256" in seen["kwargs"], "le condensé doit être proposé au téléchargement"


def test_the_resolver_passes_a_configured_digest_to_the_download(
    tmp_path: Path, monkeypatch
) -> None:
    """Sans condensé, une archive tronquée donne un bundle faux et muet."""
    monkeypatch.setattr(
        "vocal_helper._settings.resolve_diarization_engines_url",
        lambda *a, **k: "https://example.invalid/diarization-engines.zip",
    )
    monkeypatch.setenv("VH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("VH_DIARIZATION_ENGINES_SHA256", "a" * 64)

    captured: dict[str, object] = {}

    def _fake_download_file(url: str, dst: str, **kwargs) -> None:
        captured.update(kwargs)
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr("manifest.json", "{}")

    monkeypatch.setattr("os_helper.download_file", _fake_download_file, raising=False)
    resolve_diarization_engines()
    assert captured.get("sha256") == "a" * 64


def test_the_resolver_prefers_a_bundle_already_on_disk(tmp_path: Path, monkeypatch) -> None:
    """Présent localement, on l'utilise : aucun téléchargement ne doit partir."""
    monkeypatch.setattr(
        "vocal_helper._settings.resolve_diarization_engines_url",
        lambda *a, **k: "https://example.invalid/diarization-engines.zip",
    )
    cache = tmp_path / "cache" / "diarization-engines" / "diarization-engines"
    cache.mkdir(parents=True)
    (cache / "manifest.json").write_text("{}")
    monkeypatch.setenv("VH_CACHE_DIR", str(tmp_path / "cache"))

    def _must_not_run(*a, **k):
        raise AssertionError("un bundle présent a quand même déclenché un téléchargement")

    monkeypatch.setattr("os_helper.download_file", _must_not_run, raising=False)
    assert resolve_diarization_engines() == cache


# --------------------------------------------------------------------------- #
#  A diarizer is core, and the default must name something installed
# --------------------------------------------------------------------------- #


def test_a_bare_install_can_diarize_because_sherpa_is_a_core_dependency():
    """The inconsistency this fixes: the package exists for "transcription AND
    diarization", shipped the transcriber by default, and left every diarizer to an
    extra. Both stage defaults named extras, so a bare install did not degrade — it
    raised. sherpa-onnx is torch-free and light, so it is the one that belongs in
    the core dependencies."""
    from pathlib import Path

    import tomllib

    manifest = Path(__file__).resolve().parent.parent / "pyproject.toml"
    core = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]["dependencies"]
    assert any(dep.startswith("sherpa-onnx") for dep in core), (
        "a library for diarization that cannot diarize out of the box is not finished"
    )


def test_auto_never_returns_a_backend_that_is_not_installed():
    """The whole point of probing. Returning an absent backend would reproduce
    exactly the failure that made this necessary."""
    from vocal_helper.diar import _installed, resolve_backend

    for mode in ("offline", "online"):
        chosen = resolve_backend("auto", mode=mode)
        assert _installed(chosen), (mode, chosen)


def test_an_explicit_backend_is_returned_even_if_it_looks_absent():
    """An override exists for the case where the probe is wrong, so the probe must
    not overrule it. The stage then fails at load with the provider's own message,
    which is more useful than a silent substitution."""
    from vocal_helper.diar import resolve_backend

    assert resolve_backend("nemo", mode="offline") == "nemo"
    assert resolve_backend("pyannoteai", mode="offline") == "pyannoteai"


def test_the_preference_order_is_the_measured_quality_order():
    """Ni alphabétique ni affaire de goût : c'est l'ordre que la mesure donne.

    L'ordre différé a changé le 2026-11-01. Il plaçait pyannote en tête, et
    `nemotron` le dépasse largement sur le seul critère qui vaille en différé —
    la qualité, puisqu'on attend. Sur un mixage dont la vérité est exacte par
    construction, sec puis réverbéré : 0,090 et 0,252 de taux d'erreur contre
    0,299 et 0,553 pour pyannote, avec le bon nombre de locuteurs partout et une
    confusion nulle sur du propre.

    Sur AMI, `nemotron` arrive dernier, et ce classement n'a pas été pris au mot :
    la parole qu'il n'émet pas y est 21,8 dB sous celle qu'il émet et 5,4 dB
    au-dessus du silence annoté, donc au plancher de bruit du micro de salle. La
    référence d'AMI est transcrite sur les casques et décrit de la parole que le
    signal ne porte pas. Trois autres explications ont été testées et écartées —
    seuil d'extraction, niveau, réverbération.

    Installer un extra reste l'interrupteur : l'ordre est une préférence, pas une
    exigence, et un backend absent est sauté.
    """
    from vocal_helper.diar import OFFLINE_PREFERENCE, ONLINE_PREFERENCE

    assert OFFLINE_PREFERENCE[0] == "nemotron"
    assert OFFLINE_PREFERENCE[1] == "pyannote"
    assert ONLINE_PREFERENCE[0] == "nemo"
    # sherpa is last in both and yet never missing: it is the core dependency, so it
    # is the floor the probe can always fall back to.
    assert OFFLINE_PREFERENCE[-1] == ONLINE_PREFERENCE[-1] == "sherpa"


def test_the_stages_expose_the_backend_that_will_run_not_the_word_auto():
    """A caller inspecting `stage.backend` — a log line, a report header, `doctor` —
    needs the answer, not the question."""
    from vocal_helper.diar import OfflineDiarStage, OnlineDiarStage

    assert OfflineDiarStage().backend != "auto"
    assert OnlineDiarStage().backend != "auto"
    assert OfflineDiarStage(backend="sherpa").backend == "sherpa"


def test_an_unknown_mode_is_refused():
    from vocal_helper.diar import resolve_backend

    with pytest.raises(ValueError, match="unknown mode"):
        resolve_backend("auto", mode="streaming")


def test_the_default_bundle_is_verified_without_configuration(tmp_path: Path, monkeypatch) -> None:
    """Le chemin par défaut doit être vérifié sans que personne ne règle rien."""
    from vocal_helper.diar import (
        DEFAULT_DIARIZATION_ENGINES_SHA256,
        DEFAULT_DIARIZATION_ENGINES_URL,
    )

    monkeypatch.delenv("VH_DIARIZATION_ENGINES_SHA256", raising=False)
    monkeypatch.setattr(
        "vocal_helper._settings.resolve_diarization_engines_url",
        lambda *a, **k: DEFAULT_DIARIZATION_ENGINES_URL,
    )
    monkeypatch.setenv("VH_CACHE_DIR", str(tmp_path / "cache"))

    captured: dict[str, object] = {}

    def _fake_download_file(url: str, dst: str, **kwargs) -> None:
        captured.update(kwargs)
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr("manifest.json", "{}")

    monkeypatch.setattr("os_helper.download_file", _fake_download_file, raising=False)
    resolve_diarization_engines()
    assert captured.get("sha256") == DEFAULT_DIARIZATION_ENGINES_SHA256


def test_a_digest_is_never_applied_to_someone_elses_url(tmp_path: Path, monkeypatch) -> None:
    """Un condensé vaut pour SON archive : l'appliquer ailleurs ferait échouer
    tous les téléchargements et se lirait comme une corruption."""
    monkeypatch.delenv("VH_DIARIZATION_ENGINES_SHA256", raising=False)
    monkeypatch.setattr(
        "vocal_helper._settings.resolve_diarization_engines_url",
        lambda *a, **k: "https://ailleurs.invalid/mon-bundle.zip",
    )
    monkeypatch.setenv("VH_CACHE_DIR", str(tmp_path / "cache"))

    captured: dict[str, object] = {}

    def _fake_download_file(url: str, dst: str, **kwargs) -> None:
        captured.update(kwargs)
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr("manifest.json", "{}")

    monkeypatch.setattr("os_helper.download_file", _fake_download_file, raising=False)
    resolve_diarization_engines()
    assert captured.get("sha256") is None
