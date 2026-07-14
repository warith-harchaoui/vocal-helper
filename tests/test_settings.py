"""Unit tests for the tiny ``settings.yaml`` loader.

These run offline — no models, no network — and verify the documented
resolution order for the diarization-engines bundle source:
explicit > ``$VH_DIARIZATION_ENGINES`` > ``engines.diarization_url``.
No HuggingFace token is involved anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vocal_helper import _settings


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    """Write a minimal ``settings.yaml`` to a temp dir and return its path."""
    p = tmp_path / "settings.yaml"
    # The canonical config: the self-hosted bundle URL under ``engines``.
    p.write_text(
        "# leading comment\n"
        "engines:\n"
        "  diarization_url: https://host/bundle.zip  # inline comment\n"
        "  other: 'quoted value'\n",
        encoding="utf-8",
    )
    return p


def test_parse_minimal_yaml_extracts_nested_value(yaml_file: Path) -> None:
    """The hand-rolled parser reads the two-level ``section: {key: value}``."""
    parsed = _settings._parse_minimal_yaml(yaml_file.read_text())
    assert parsed["engines"]["diarization_url"] == "https://host/bundle.zip"
    # Quotes around values are stripped.
    assert parsed["engines"]["other"] == "quoted value"


def test_resolve_explicit_wins_over_env_and_file(
    monkeypatch: pytest.MonkeyPatch,
    yaml_file: Path,
) -> None:
    """An explicit argument beats both the env var and the settings file."""
    monkeypatch.setenv("VOCAL_HELPER_SETTINGS", str(yaml_file))
    monkeypatch.setenv("VH_DIARIZATION_ENGINES", "from-env")
    assert _settings.resolve_diarization_engines_url("from-arg") == "from-arg"


def test_resolve_env_wins_over_file(
    monkeypatch: pytest.MonkeyPatch,
    yaml_file: Path,
) -> None:
    """The ``$VH_DIARIZATION_ENGINES`` env var beats the settings file."""
    monkeypatch.setenv("VOCAL_HELPER_SETTINGS", str(yaml_file))
    monkeypatch.setenv("VH_DIARIZATION_ENGINES", "from-env")
    assert _settings.resolve_diarization_engines_url() == "from-env"


def test_resolve_falls_back_to_settings_file(
    monkeypatch: pytest.MonkeyPatch,
    yaml_file: Path,
) -> None:
    """With no arg and no env var, the settings-file value is used."""
    monkeypatch.setenv("VOCAL_HELPER_SETTINGS", str(yaml_file))
    monkeypatch.delenv("VH_DIARIZATION_ENGINES", raising=False)
    assert _settings.resolve_diarization_engines_url() == "https://host/bundle.zip"


def test_resolve_returns_none_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No env, no settings file → ``None`` (cwd + repo-root lookup neutralised)."""
    monkeypatch.delenv("VH_DIARIZATION_ENGINES", raising=False)
    monkeypatch.delenv("VOCAL_HELPER_SETTINGS", raising=False)
    monkeypatch.chdir(tmp_path)  # empty cwd
    # Bypass the package-root fallback by stubbing settings_path.
    monkeypatch.setattr(_settings, "settings_path", lambda: None)
    assert _settings.resolve_diarization_engines_url() is None
