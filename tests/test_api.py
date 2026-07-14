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
