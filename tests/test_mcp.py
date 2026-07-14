"""
Smoke test for the MCP surface (``vocal_helper.mcp``).

Verifies that the MCP wrapper around the FastAPI app imports without
error, exposes the underlying FastAPI ``app`` object, and that the
``mcp`` handler is attached. Full MCP protocol round-trips belong to
a separate integration suite once the client tooling is stable in CI.

Usage Example
-------------
>>> #   pytest tests/test_mcp.py

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import pytest

# fastapi_mcp lives in the ``[mcp]`` optional extra — skip cleanly if absent.
pytest.importorskip("fastapi_mcp")


def test_mcp_module_imports_and_exposes_app() -> None:
    """The MCP module must import and re-expose the FastAPI app + mcp handler."""
    # A bare import is the real smoke test : the fastapi_mcp wrapper wires
    # itself onto the FastAPI app at import time, so a broken mount blows up here.
    from vocal_helper import mcp as mcp_module

    # ``app`` is re-exported so callers can mount it ; ``mcp`` is the protocol
    # handler. Both must survive the wrapping.
    assert hasattr(mcp_module, "app"), "vocal_helper.mcp must re-expose `app`."
    assert hasattr(mcp_module, "mcp"), "vocal_helper.mcp must expose the `mcp` handler."


def test_main_entrypoint_is_callable() -> None:
    """The ``vocal-helper-mcp`` console entry point should be a callable."""
    # ``main`` is what the pyproject console_scripts entry binds to — assert it
    # exists and is callable so a packaging regression fails here, not on launch.
    from vocal_helper.mcp import main

    assert callable(main)
