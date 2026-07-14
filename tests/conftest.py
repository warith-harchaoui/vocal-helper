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
    # Honour the explicit override env var — that path is set *by the test*
    # itself, so it stays deterministic.
    override = os.environ.get("VOCAL_HELPER_SETTINGS")
    if override:
        p = Path(override).expanduser()
        # A non-existent override is treated as "unset" rather than an error.
        if p.is_file():
            return p
    # No override → return None instead of walking cwd / repo-root, which is
    # the whole point : an ambient settings.yaml must never leak into a test.
    return None


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ambient HF/settings inputs from every test."""
    # A developer's shell may export HF_TOKEN ; drop it so token-resolution
    # tests see the CI-like "no token" world.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Clear any explicit settings pointer so each test starts from a clean
    # slate and opts back in only when it needs to.
    monkeypatch.delenv("VOCAL_HELPER_SETTINGS", raising=False)
    # Swap the resolver for the env-only variant : suppresses the implicit
    # cwd / repo-root lookup for the duration of the test.
    monkeypatch.setattr(_settings, "settings_path", _env_only_settings_path)
