"""Shared pytest fixtures and isolation hooks.

The package looks for ``settings.yaml`` in the current working
directory and then next to the repo root. On a developer's machine
that file may already exist with a real HF token ; on CI it won't.
Either way, the resolver tests need a *deterministic* environment to
assert the documented order — explicit > env > file.

This conftest replaces :func:`vocal_helper._settings.settings_path`
with an env-only variant for every test : if a test sets
``$VOCAL_HELPER_SETTINGS`` it still resolves through that override,
but the implicit cwd / repo-root lookup is suppressed so an ambient
``settings.yaml`` cannot leak into a test. Ambient ``HF_TOKEN`` is
also unset.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from vocal_helper import _settings


def _env_only_settings_path() -> Path | None:
    """Mirror :func:`settings_path` but skip the cwd / repo-root fallback."""
    override = os.environ.get("VOCAL_HELPER_SETTINGS")
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return p
    return None


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ambient HF/settings inputs from every test."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("VOCAL_HELPER_SETTINGS", raising=False)
    monkeypatch.setattr(_settings, "settings_path", _env_only_settings_path)
