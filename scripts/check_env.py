"""Does the local environment satisfy what pyproject.toml declares?

The server CI installs a fresh environment from ``pyproject.toml`` on every run;
a workstation keeps one for months. The two drift, and when they do the local
gate stops predicting the server: on 2026-11-04 the ``.venv`` here carried
os-helper 2.0.1 against a declared ``>=4.1.0`` and had no sherpa-onnx at all,
so ``make preflight`` went red on a test the server passes — and the opposite
case, a stale environment green where a fresh one fails, is the one that ends
with a blocked push on the server.

This script reads the base dependencies and the requested extras, asks
``importlib.metadata`` what is installed, and names every requirement that is
missing or out of range. Stdlib plus ``packaging`` (which pip itself vendors and
every dev environment has), so it runs before anything else is trusted.

Usage::

    python scripts/check_env.py            # base + dev
    python scripts/check_env.py --extra bench --extra api
"""

from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent


def requirements(extras: list[str]) -> list[Requirement]:
    """Base dependencies plus the named extras, self-references expanded."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    optional = project.get("optional-dependencies", {})
    out: list[Requirement] = [Requirement(r) for r in project["dependencies"]]
    seen: set[str] = set()
    todo = list(extras)
    while todo:
        extra = todo.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for raw in optional.get(extra, []):
            req = Requirement(raw)
            if req.name == project["name"]:
                todo.extend(req.extras)
            else:
                out.append(req)
    return [r for r in out if r.marker is None or r.marker.evaluate()]


def problems(reqs: list[Requirement]) -> list[str]:
    """One line per requirement the environment does not meet."""
    out = []
    for req in reqs:
        try:
            have = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            out.append(f"{req}  (not installed)")
            continue
        if req.specifier and not req.specifier.contains(have, prereleases=True):
            out.append(f"{req}  (installed {have})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extra", action="append", default=None)
    args = parser.parse_args()
    extras = args.extra if args.extra is not None else ["dev"]
    bad = problems(requirements(extras))
    if bad:
        print("environment does not match pyproject.toml:", file=sys.stderr)
        for line in bad:
            print(f"  - {line}", file=sys.stderr)
        print("fix: make sync", file=sys.stderr)
        return 1
    print(f"environment matches pyproject.toml (base + {', '.join(extras)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
