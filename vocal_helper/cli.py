"""
vocal_helper.cli — backwards-compat shim.

The canonical argparse CLI lives in :mod:`vocal_helper.cli_argparse`
so it can sit next to the click twin (:mod:`vocal_helper.cli_click`)
with symmetric naming — and so both CLI surfaces share a single config
builder without drift. This module keeps the old import path
(``from vocal_helper.cli import main``) working so downstream code and
tests continue to load.

The shipped ``vocal-helper`` entry point resolves to
:func:`vocal_helper.cli_argparse.main` (see ``[project.scripts]`` in
``pyproject.toml``); everything you can do here you can do there, plus
the ``url`` and ``transcribe`` subcommands.

Usage Example
-------------
>>> from vocal_helper.cli import main
>>> # equivalent to :
>>> from vocal_helper.cli_argparse import main

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

# Re-export the public entry points so legacy imports keep working. The
# private ``_build_config`` alias is what the CLI tests reach for; it is
# the exact same builder the argparse and click surfaces call, so there
# is only one place where "CLI namespace -> PipelineConfig" is defined.
from vocal_helper.cli_argparse import (  # noqa: F401
    _build_pipeline_config as _build_config,
)
from vocal_helper.cli_argparse import (
    build_parser,
    main,
)

__all__ = ["_build_config", "build_parser", "main"]
