"""
Smoke tests for the FastAPI HTTP surface.

Only exercises endpoints that do not require whisper.cpp, pyannote, or
the LLM analyst (``/health``, plus OpenAPI schema introspection to
catch endpoint-name drift). Heavier round-trip tests belong to the
``integration`` suite where model weights are available.

Usage Example
-------------
>>> #   pytest tests/test_api.py

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import pytest

# FastAPI is in the ``[api]`` optional extra — skip cleanly otherwise.
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Yield a TestClient bound to the vocal-helper FastAPI app."""
    # Import lazily so the [api] extra is only required when this fixture runs.
    from vocal_helper.api import app

    # ``with`` triggers FastAPI startup/shutdown events around the yielded client.
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client: TestClient) -> None:
    """``/health`` should return 200 + ``{"status": "ok"}``."""
    # Cheapest liveness probe : no models touched, so it's a pure wiring check.
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_openapi_lists_expected_endpoints(client: TestClient) -> None:
    """The OpenAPI spec should list every expected route path."""
    # Introspect the generated schema instead of calling the heavy endpoints —
    # this catches route renames / drops without loading whisper or pyannote.
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    # The three routes downstream tooling relies on ; extras are allowed.
    expected = {"/health", "/transcribe", "/pipeline"}
    assert expected.issubset(set(paths.keys()))


def test_docs_endpoint_is_served(client: TestClient) -> None:
    """``/docs`` should serve the Swagger UI landing HTML."""
    r = client.get("/docs")
    assert r.status_code == 200
    # The Swagger bundle self-identifies via one of these tokens in the HTML.
    assert "swagger" in r.text.lower() or "openapi" in r.text.lower()


def test_minimal_gui_served_at_ui(client: TestClient) -> None:
    """The minimal web GUI is served same-origin at ``/ui`` in a source checkout."""
    # The API mounts webui/ at /ui only when the folder sits next to the package
    # (a repo checkout — always true in CI). A bare pip install without the folder
    # skips the mount, so treat a 404 as "not a source tree" and skip, not fail.
    from vocal_helper import api

    if not api._WEBUI_DIR.is_dir():
        pytest.skip("webui/ folder absent (bare install, not a source checkout)")
    # StaticFiles(html=True) serves index.html at the mount root.
    r = client.get("/ui/")
    assert r.status_code == 200
    # It is the self-contained local GUI page, not an API JSON error.
    assert "html" in r.text.lower()


def test_pipeline_backend_default_is_auto() -> None:
    """The ``/pipeline`` ``diar_backend`` form field must default to 'auto'."""
    # Guard against regressing to the old hardcoded 'pyannote' default that
    # bypassed the router entirely on the server side.
    import inspect

    from vocal_helper.api import pipeline

    default = inspect.signature(pipeline).parameters["diar_backend"].default
    # FastAPI wraps the default in a Form(...) marker — its ``.default`` is the value.
    assert getattr(default, "default", default) == "auto"


def test_resolve_offline_backend_routes_by_duration(monkeypatch) -> None:
    """``_resolve_offline_backend`` routes short→nemo / long→pyannote, honours overrides."""
    # Model-free: pin both availability probes so the router exercises its
    # length crossover without importing pyannote / NeMo.
    from vocal_helper import cli_argparse as cli
    from vocal_helper.api import _resolve_offline_backend

    monkeypatch.setattr(cli, "_offline_pyannote_available", lambda: True)
    monkeypatch.setattr(cli, "_offline_nemo_available", lambda: True)
    sr = 16_000
    # 45 s of audio → short/dense branch → nemo.
    assert _resolve_offline_backend("auto", 45 * sr, sr) == "nemo"
    # 1800 s of audio → long-form branch → pyannote.
    assert _resolve_offline_backend("auto", 1800 * sr, sr) == "pyannote"
    # Unknown sample rate collapses to unknown duration → robust pyannote branch.
    assert _resolve_offline_backend("auto", 45 * sr, 0) == "pyannote"
    # An explicit backend is honoured verbatim, router untouched.
    assert _resolve_offline_backend("sherpa", 45 * sr, sr) == "sherpa"
