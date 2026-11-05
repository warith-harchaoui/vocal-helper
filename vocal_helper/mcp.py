"""
Vocal Helper — Model Context Protocol (MCP) surface.

Adapter that exposes the FastAPI app defined in :mod:`vocal_helper.api`
as MCP tools so any MCP-aware host (agent runtimes, IDE
integrations, custom shells) can call ``transcribe`` / ``pipeline`` as
first-class tools. Uses :mod:`fastapi_mcp`
(https://github.com/tadata-org/fastapi_mcp) — one line wraps the whole
existing HTTP surface, so we never duplicate the route definitions.

Install the extra to pull in ``fastapi-mcp``::

    pip install 'vocal-helper[api,mcp]'

Then run the MCP server::

    vocal-helper-mcp                 # entry point (see pyproject)
    # or, equivalently:
    python -m vocal_helper.mcp

Usage Example
-------------
>>> # Register http://127.0.0.1:8000/mcp in your client. It publishes the
>>> # routes under fastapi-mcp's operation ids:
>>> #   transcribe_transcribe_post / pipeline_pipeline_post
>>> #   health_health_get / gui_gui_get
>>> # …with the same argument names as the FastAPI routes.

Author
------
Warith Harchaoui, Ph.D. , https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

try:
    from fastapi_mcp import FastApiMCP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The MCP surface requires the [mcp] extra. "
        "Install with: pip install 'vocal-helper[api,mcp]'"
    ) from exc

# Reuse the exact same FastAPI app — MCP is a thin wrapper on top.
from vocal_helper.api import app

# ``FastApiMCP`` mounts an MCP endpoint on the existing FastAPI app; we
# store the wrapped instance at module scope so downstream code (tests,
# ASGI runners) can access both the FastAPI app and the MCP handler.
mcp = FastApiMCP(
    app,
    name="vocal-helper",
    description=(
        "Vocal Helper MCP tools: transcribe an audio file, or run the full "
        "offline pipeline (VAD + diarization + STT + optional Gemma summary)."
    ),
)
# Attach the MCP endpoint to the FastAPI app. Newer fastapi-mcp releases
# split ``mount()`` into transport-specific ``mount_http()`` (recommended)
# and ``mount_sse()``. Fall back to the legacy ``mount()`` on older
# versions so users can install a range of ``fastapi-mcp`` versions.
if hasattr(mcp, "mount_http"):
    mcp.mount_http()
else:  # pragma: no cover — legacy fastapi-mcp
    mcp.mount()


def main(argv: list[str] | None = None) -> None:
    """
    Entry point for the ``vocal-helper-mcp`` console script.

    Boots the FastAPI app (which now serves both the ``/…`` HTTP routes
    and the MCP endpoint) with ``uvicorn`` in single-worker mode. Meant
    for local / container usage; behind a real load balancer use
    ``uvicorn`` / ``gunicorn`` directly. The bind comes from ``--host`` /
    ``--port`` or ``VOCAL_HELPER_HOST`` / ``VOCAL_HELPER_PORT``. A click
    command, so ``--help`` answers instead of starting the server.
    """
    import click

    @click.command(context_settings={"help_option_names": ["-h", "--help"]})
    @click.option("--host", envvar="VOCAL_HELPER_HOST", default="0.0.0.0", show_default=True)
    @click.option("--port", envvar="VOCAL_HELPER_PORT", default=8000, type=int, show_default=True)
    def serve(host: str, port: int) -> None:
        """Serve the vocal-helper HTTP API with its MCP endpoint."""
        import uvicorn

        # Single worker keeps whisper.cpp + pyannote model instances in a
        # shared process (no reload storms across workers).
        uvicorn.run(app, host=host, port=port, workers=1)

    serve.main(args=argv, prog_name="vocal-helper-mcp")


if __name__ == "__main__":  # pragma: no cover
    main()
