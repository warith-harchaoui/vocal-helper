"""Unit tests for the tiny ``settings.yaml`` loader.

These run offline — no models, no network — and verify the documented
resolution order : explicit > ``$HF_TOKEN`` > ``secrets.hf_token``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vocal_helper import _settings


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    """Write a minimal ``settings.yaml`` to a temp dir and return its path."""
    p = tmp_path / "settings.yaml"
    p.write_text(
        "# leading comment\n"
        "secrets:\n"
        "  hf_token: hf_REAL  # inline comment after value\n"
        "  other: 'quoted value'\n",
        encoding="utf-8",
    )
    return p


def test_parse_minimal_yaml_extracts_nested_value(yaml_file: Path) -> None:
    parsed = _settings._parse_minimal_yaml(yaml_file.read_text())
    assert parsed["secrets"]["hf_token"] == "hf_REAL"
    # Quotes around values are stripped.
    assert parsed["secrets"]["other"] == "quoted value"


def test_resolve_explicit_wins_over_env_and_file(
    monkeypatch: pytest.MonkeyPatch, yaml_file: Path,
) -> None:
    monkeypatch.setenv("VOCAL_HELPER_SETTINGS", str(yaml_file))
    monkeypatch.setenv("HF_TOKEN", "hf_FROM_ENV")
    assert _settings.resolve_hf_token("hf_FROM_ARG") == "hf_FROM_ARG"


def test_resolve_env_wins_over_file(
    monkeypatch: pytest.MonkeyPatch, yaml_file: Path,
) -> None:
    monkeypatch.setenv("VOCAL_HELPER_SETTINGS", str(yaml_file))
    monkeypatch.setenv("HF_TOKEN", "hf_FROM_ENV")
    assert _settings.resolve_hf_token() == "hf_FROM_ENV"


def test_resolve_falls_back_to_settings_file(
    monkeypatch: pytest.MonkeyPatch, yaml_file: Path,
) -> None:
    monkeypatch.setenv("VOCAL_HELPER_SETTINGS", str(yaml_file))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert _settings.resolve_hf_token() == "hf_REAL"


def test_placeholder_value_is_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Copying ``settings.yaml.example`` shouldn't masquerade as real auth."""
    p = tmp_path / "settings.yaml"
    p.write_text("secrets:\n  hf_token: hf_XXXX\n", encoding="utf-8")
    monkeypatch.setenv("VOCAL_HELPER_SETTINGS", str(p))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert _settings.resolve_hf_token() is None


def test_resolve_returns_none_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No env, no settings file → ``None`` (cwd + repo-root lookup neutralised)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("VOCAL_HELPER_SETTINGS", raising=False)
    monkeypatch.chdir(tmp_path)  # empty cwd
    # Bypass the package-root fallback by stubbing settings_path.
    monkeypatch.setattr(_settings, "settings_path", lambda: None)
    assert _settings.resolve_hf_token() is None
