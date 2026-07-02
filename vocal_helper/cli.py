"""
vocal_helper.cli — backwards-compat shim.

The canonical argparse CLI now lives in :mod:`vocal_helper.cli_argparse`
so it can sit next to the click twin (:mod:`vocal_helper.cli_click`),
the FastAPI HTTP surface (:mod:`vocal_helper.api`), and the MCP surface
(:mod:`vocal_helper.mcp`) with symmetric naming. This module keeps the
old import path (`from vocal_helper.cli import main`) working so
downstream code and tests continue to load.

Usage Example
-------------
>>> from vocal_helper.cli import main
>>> # equivalent to :
>>> from vocal_helper.cli_argparse import main

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

# Re-export the public entry points so legacy imports keep working.
from vocal_helper.cli_argparse import (  # noqa: F401
    _build_pipeline_config as _build_config,
)
from vocal_helper.cli_argparse import (
    build_parser,
    main,
)

__all__ = ["_build_config", "build_parser", "main"]
