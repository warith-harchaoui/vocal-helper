"""
vocal_helper._settings
======================

Local configuration loader for ``settings.yaml``.

The library reads its configuration from a ``settings.yaml`` file
checked in *only* as ``settings.yaml.example`` — the real file is
git-ignored and lives next to the package root (or the current
working directory). The schema is intentionally tiny :

.. code-block:: yaml

    # The self-hosted model bundle — the ONLY config the project needs.
    # It removes every HuggingFace dependency (no token required).
    engines:
      diarization_url: https://…/diarization-engines.zip

Public entry points
-------------------

* :func:`settings_path` — return the resolved file path, or ``None``.
* :func:`load_settings` — parse it into a two-level ``dict``.
* :func:`resolve_diarization_engines_url` — implement the documented
  source-resolution order for the self-hosted model bundle so every call
  site (CLI, library classes, examples) behaves identically.

Resolution order for the engines bundle (highest priority first)
----------------------------------------------------------------

1. The explicit value passed by the caller.
2. The ``VH_DIARIZATION_ENGINES`` environment variable (URL or local dir).
3. ``engines.diarization_url`` from ``settings.yaml``.

A missing file or missing key returns ``None`` ; the diar backends then
fall back to their built-in default URL. **No HuggingFace token is used
anywhere** — the bundle carries every weight the project needs.

YAML support
------------

The loader is hand-rolled (no PyYAML dependency) and tolerates *only*
the documented two-level schema : top-level ``section:`` headers and
indented ``key: value`` pairs. Inline ``#`` comments are stripped ;
single- or double-quoted values are unquoted. Anything deeper or
list-shaped is ignored — keep ``settings.yaml`` flat.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "load_settings",
    "resolve_diarization_engines_url",
    "settings_path",
]


def _strip_comment(value: str) -> str:
    """Drop the first un-quoted ``#`` comment, keeping ``#`` inside quotes."""
    # Track quote context by hand — a URL like ``https://…#frag`` inside
    # a quoted value must survive, so a naive ``split("#")`` won't do.
    in_single = False
    in_double = False
    for i, ch in enumerate(value):
        # A single quote only toggles state when we're NOT inside a
        # double-quoted run (and vice-versa) ; that's how the two quote
        # styles nest without clobbering each other.
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        # A ``#`` outside every quote starts the comment — truncate here.
        elif ch == "#" and not in_single and not in_double:
            return value[:i]
    return value


def _unquote(value: str) -> str:
    """Strip one matching pair of surrounding quotes, if present."""
    value = value.strip()
    # Only peel quotes when both ends match the SAME quote char — a value
    # like ``"a'`` (mismatched) is left verbatim rather than corrupted.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_minimal_yaml(text: str) -> dict[str, dict[str, str]]:
    """Parse a flat two-level ``section: { key: value }`` document.

    Only the structure used by ``settings.yaml`` is supported ; deeper
    nesting or sequences silently fall through. The return type is
    always ``{section: {key: value}}`` so callers can use ``.get``
    safely on either level.
    """
    out: dict[str, dict[str, str]] = {}
    # ``current`` names the section we're accumulating keys into ; ``None``
    # means we're at top level (or inside an unrecognised construct).
    current: str | None = None
    for raw_line in text.splitlines():
        # Strip inline comments and trailing whitespace up front so the
        # structural checks below only ever see meaningful content.
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        # Zero indentation ⇒ a top-level line, i.e. a new section header.
        if not line.startswith((" ", "\t")):
            # New top-level section header — ``section:``.
            if line.endswith(":"):
                current = line[:-1].strip()
                out.setdefault(current, {})
            else:
                # Top-level scalar — not part of the schema, skip.
                current = None
            continue
        # Indented line — only meaningful inside a known section. An
        # indented key that appears before any header is orphaned; drop it.
        if current is None:
            continue
        stripped = line.lstrip()
        # A ``key: value`` pair needs a colon ; anything else isn't schema.
        if ":" not in stripped:
            continue
        # ``partition`` (not ``split``) so a value that itself contains a
        # colon — e.g. an ``https://`` URL — keeps its right-hand side intact.
        key, _, value = stripped.partition(":")
        out[current][key.strip()] = _unquote(value)
    return out


def settings_path() -> Path | None:
    """Return the resolved ``settings.yaml`` path, or ``None`` if absent.

    Search order :

    1. ``$VOCAL_HELPER_SETTINGS`` if set and pointing at an existing
       file. Lets tests and unusual deploys pin a specific file.
    2. ``settings.yaml`` in the current working directory.
    3. ``settings.yaml`` next to the installed package — i.e. the
       repo root in editable installs.

    The example file (``settings.yaml.example``) is *not* searched ;
    users must copy it to ``settings.yaml`` to opt in.
    """
    # (1) Explicit override wins — but only if it actually resolves to a
    # file, so a stale env var can't shadow a valid on-disk settings file.
    override = os.environ.get("VOCAL_HELPER_SETTINGS")
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return p
    # (2) CWD first, then (3) the repo root beside the package — CWD is
    # checked first so a per-project settings.yaml beats a global one.
    candidates = [
        Path.cwd() / "settings.yaml",
        Path(__file__).resolve().parent.parent / "settings.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    # Nothing found — callers treat ``None`` as "use built-in defaults".
    return None


def load_settings() -> dict[str, dict[str, str]]:
    """Read ``settings.yaml`` and return its parsed mapping.

    Returns an empty ``dict`` when the file is missing or unreadable,
    so callers can chain ``.get("engines", {}).get("diarization_url")``
    without guarding for ``None``.
    """
    p = settings_path()
    if p is None:
        return {}
    # Swallow read errors (permission / race with deletion) into an empty
    # mapping — config is optional, so an unreadable file must not crash
    # a pipeline that would otherwise fall back to built-in defaults.
    try:
        return _parse_minimal_yaml(p.read_text(encoding="utf-8"))
    except OSError:
        return {}


# Placeholder shipped in ``settings.yaml.example`` — treated as "unset".
_PLACEHOLDER_ENGINES = frozenset({"", "https://example.com/diarization-engines.zip"})


def resolve_diarization_engines_url(explicit: str | None = None) -> str | None:
    """Return the diarization-engines bundle source (URL or local dir).

    The bundle carries every model weight the project needs — the offline
    pyannote pipeline, NeMo Sortformer, the online ``pyannote/embedding``
    embedder and SpeechBrain VoxLingua107 — so nothing is fetched from
    HuggingFace. This is the *only* configuration the library requires.

    Parameters
    ----------
    explicit : str, optional
        A source supplied directly by the caller. Takes precedence.

    Returns
    -------
    str or None
        A URL to ``diarization-engines.zip`` or a local bundle directory,
        or ``None`` when no source is configured (callers then use their
        own built-in default).

    Notes
    -----
    Resolution order (highest priority first):

    1. ``explicit`` argument.
    2. ``$VH_DIARIZATION_ENGINES`` environment variable (URL or local dir).
    3. ``engines.diarization_url`` in ``settings.yaml``.
    """
    # (1) Caller-supplied value always wins.
    if explicit:
        return explicit
    # (2) Environment override — used by tests to point at a local bundle.
    env = os.environ.get("VH_DIARIZATION_ENGINES")
    if env:
        return env
    # (3) The documented settings.yaml key — the canonical config source.
    url = load_settings().get("engines", {}).get("diarization_url")
    if url and url not in _PLACEHOLDER_ENGINES:
        return url
    return None
