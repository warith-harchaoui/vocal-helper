"""
vocal_helper.credentials
========================

Credential resolution for the **paid cloud backends**, and nothing else.

vocal-helper is local-first: the two local modes need no account, no key and no
network. This module exists only so the two *paid* modes can find their keys,
and it is written so that **a missing key means the mode is unavailable**, never
that something silently degrades or silently bills.

Why a module of its own
-----------------------
Three properties are easier to guarantee in one small place than scattered
across the cloud stages.

1. **Opt-in is structural.** :func:`credential` returns ``""`` when nothing is
   configured. Every cloud stage raises on an empty key rather than guessing, so
   there is no code path where a key arriving by accident turns a free run into a
   billed one.
2. **One resolution order, written down.** Environment variable first (that is
   how continuous integration and one-off experiments inject a key), then the
   writable overlay, then ``settings.yaml``. The overlay wins over the file
   because the file is richly commented and meant to be read by humans, while the
   overlay is machine-written by a preferences panel; rewriting the commented
   file to store a secret would destroy its comments.
3. **Reporting without leaking.** :func:`credentials_status` answers "which paid
   services could run?" with booleans only. A status endpoint, a ``doctor``
   command or a log line can say *configured* / *not configured* without ever
   putting a key where it might be copied.

The names are the ones the rest of the suite already uses, so a machine
configured for another project's paid path is configured for this one too.

Resolution order
----------------
For a name ``GLADIA_API_KEY``:

1. ``os.environ["GLADIA_API_KEY"]`` (also tried lower-cased);
2. the ``credentials:`` section of the ``credentials.yaml`` overlay;
3. the ``credentials:`` section of ``settings.yaml``.

Both files carry the **same shape** — a ``credentials:`` section of indented
``NAME: value`` pairs — so one parser reads both and there is no second format to
learn or to keep in sync. They sit next to the working directory or at the
package's repository root, exactly where
:func:`vocal_helper._settings.settings_path` already looks, and **neither may be
committed**.

Usage example
-------------
>>> import os
>>> os.environ["GLADIA_API_KEY"] = "k-123"
>>> credential("GLADIA_API_KEY")
'k-123'
>>> credential("NOT_CONFIGURED_ANYWHERE")
''
>>> del os.environ["GLADIA_API_KEY"]
"""

from __future__ import annotations

import os
from pathlib import Path

from vocal_helper._settings import _parse_minimal_yaml, settings_path

__all__ = [
    "PAID_SERVICES",
    "credential",
    "credentials_status",
    "overlay_path",
    "set_credentials",
]


# Which keys unlock which paid service. Grouped so a caller can ask "can the paid
# offline mode run?" instead of knowing that it happens to need two different
# providers — a fact that belongs here and not in the caller.
PAID_SERVICES: dict[str, list[str]] = {
    # Paid OFFLINE: Gladia solaria-3 for the words, pyannoteAI for the speakers.
    "cloud_offline": ["GLADIA_API_KEY", "PYANNOTEAI_API_KEY"],
    # Paid LIVE: Gladia solaria-1 (the only model the live endpoint accepts) and
    # pyannoteAI Live-1. Same keys, different endpoints.
    "cloud_live": ["GLADIA_API_KEY", "PYANNOTEAI_API_KEY"],
    # ElevenLabs Scribe is a selectable third transcriber, never a default.
    "cloud_elevenlabs": ["ELEVENLABS_API_KEY"],
}


def overlay_path() -> Path:
    """Locate the writable ``credentials.yaml`` overlay.

    The overlay is where a preferences panel persists keys a user typed, so the
    commented ``settings.yaml`` is never rewritten. It carries the same
    ``credentials:`` section shape as ``settings.yaml``. Looked up in the current
    working directory first, then beside the package's repository root.

    Returns
    -------
    pathlib.Path
        The overlay path. It may not exist yet; :func:`set_credentials` creates it.

    Examples
    --------
    >>> overlay_path().name
    'credentials.yaml'
    """
    here = Path.cwd() / "credentials.yaml"
    if here.is_file():
        return here
    return (Path(__file__).resolve().parent.parent / "credentials.yaml").resolve()


def _stored() -> dict[str, str]:
    """Merge the stored credentials, overlay winning over ``settings.yaml``.

    Returns
    -------
    dict of str
        Lower-cased names mapped to their values. Empty when nothing is stored.
    """
    merged: dict[str, str] = {}

    path = settings_path()
    if path is not None:
        try:
            section = _parse_minimal_yaml(path.read_text(encoding="utf-8")).get("credentials", {})
        except OSError:
            section = {}
        merged.update({str(k).lower(): str(v) for k, v in section.items() if v})

    overlay = overlay_path()
    if overlay.is_file():
        try:
            section = _parse_minimal_yaml(overlay.read_text(encoding="utf-8")).get(
                "credentials", {}
            )
        except OSError:
            section = {}
        merged.update({str(k).lower(): str(v) for k, v in section.items() if v})

    return merged


def credential(name: str, default: str = "") -> str:
    """Resolve one credential, environment first.

    Parameters
    ----------
    name : str
        The credential name, conventionally upper-case
        (``"GLADIA_API_KEY"``, ``"PYANNOTEAI_API_KEY"``, ``"ELEVENLABS_API_KEY"``).
    default : str, optional
        Returned when nothing is configured anywhere. Defaults to ``""``, which
        every cloud stage treats as "this mode is unavailable".

    Returns
    -------
    str
        The value, or ``default``. Never ``None``, so a caller can test it with a
        plain truth check.

    Examples
    --------
    >>> import os
    >>> os.environ["PYANNOTEAI_API_KEY"] = "p-456"
    >>> credential("PYANNOTEAI_API_KEY")
    'p-456'
    >>> del os.environ["PYANNOTEAI_API_KEY"]
    >>> credential("PYANNOTEAI_API_KEY", default="absent")
    'absent'
    """
    for key in (name, name.upper(), name.lower()):
        value = os.environ.get(key)
        if value:
            return value
    stored = _stored().get(name.lower())
    return str(stored) if stored else default


def credentials_status() -> dict[str, bool]:
    """Report which paid services are configured, without revealing any value.

    Returns
    -------
    dict of bool
        One entry per name in :data:`PAID_SERVICES`, plus one per service that is
        ``True`` only when **every** key that service needs is present. A service
        with a single missing key reads as unavailable, which is the honest answer:
        the paid offline mode cannot run on half its providers.

    Examples
    --------
    >>> status = credentials_status()
    >>> set(status) >= {"cloud_offline", "GLADIA_API_KEY"}
    True
    >>> all(isinstance(v, bool) for v in status.values())
    True
    """
    names = sorted({n for keys in PAID_SERVICES.values() for n in keys})
    status: dict[str, bool] = {name: bool(credential(name)) for name in names}
    for service, keys in PAID_SERVICES.items():
        status[service] = all(status.get(k, False) for k in keys)
    return status


def set_credentials(mapping: dict[str, str]) -> Path:
    """Persist ``NAME: value`` pairs into the writable overlay.

    Merges into any existing overlay, so a partial save keeps untouched keys, and
    **skips empty values** so a user interface can send masked placeholders back
    without wiping a stored key. Never touches ``settings.yaml``.

    Parameters
    ----------
    mapping : dict of str
        The pairs to store, e.g. ``{"GLADIA_API_KEY": "..."}``. Empty or
        whitespace-only values are ignored.

    Returns
    -------
    pathlib.Path
        The overlay path written.

    Examples
    --------
    >>> import tempfile, os, pathlib
    >>> d = tempfile.mkdtemp(); cwd = os.getcwd(); os.chdir(d)
    >>> _ = pathlib.Path("credentials.yaml").write_text("")
    >>> p = set_credentials({"GLADIA_API_KEY": "written", "BLANK_KEY": "  "})
    >>> "GLADIA_API_KEY: written" in p.read_text()
    True
    >>> "BLANK_KEY" in p.read_text()
    False
    >>> credential("GLADIA_API_KEY")
    'written'
    >>> os.chdir(cwd)
    """
    path = overlay_path()
    current: dict[str, str] = {}
    if path.is_file():
        try:
            section = _parse_minimal_yaml(path.read_text(encoding="utf-8")).get("credentials", {})
            current.update({str(k): str(v) for k, v in section.items()})
        except OSError:
            current = {}

    for name, value in mapping.items():
        if str(value).strip():
            current[str(name)] = str(value).strip()

    body = "".join(f"  {name}: {value}\n" for name, value in sorted(current.items()))
    path.write_text(
        "# vocal-helper — paid-backend credentials. Machine-written, never committed.\n"
        "# Edited by a preferences panel; settings.yaml stays human-owned and commented.\n"
        "# An environment variable of the same name overrides anything below.\n"
        "credentials:\n" + body,
        encoding="utf-8",
    )
    return path
