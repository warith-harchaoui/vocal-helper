"""
vocal_helper.wallet
===================

Never pay twice for the same audio.

The two paid modes place billed calls to hosted services. This module is the
ledger in front of them: a paid call is memoised, across processes and across
restarts, on a key derived from the **audio bytes themselves**, so re-running a
build, re-extracting the same audio or re-rendering a report costs nothing the
second time.

Extracted from ``notes_helper.cloud_transcribe`` (same author), where the
reasoning below was learned the expensive way; it belongs in the library that
owns the paid calls, not in a product on top of it.

Why the key is the audio's bytes
--------------------------------
The obvious ledger key is ``path:size:mtime``. It is wrong here, and the failure
is silent.

Re-extracting the exact same audio — a rebuild, a reprocess, a second pass with a
different glossary — rewrites the file with a fresh modification time while the
bytes are identical. A ``size:mtime`` key would change, miss the ledger, and bill
the call again. Worse: when the quota is spent, the miss does not merely cost
money, it fails over and the run quietly produces a different (local) result.

Hashing the bytes makes the key survive re-extraction: identical audio means an
identical key, which means a hit.

Why most keyword arguments are excluded from the key
----------------------------------------------------
Only the arguments that genuinely change the *output* belong in the key.

- **Transcription** keys on the audio alone. A changed glossary or keyterm bias, a
  language relabel, a different prompt: none of these justify paying to transcribe
  the same bytes a second time.
- **Diarization** keys on the audio plus an explicit speaker count, when one is
  forced. Auto-detection is a pure function of the audio, but a run with a forced
  count must not silently reuse the auto-detected result, so the two are cached
  separately.
- **A transcriber that bundles diarization** (ElevenLabs Scribe does; Gladia does
  not) therefore keys on the diarization arguments too.

Escape hatch
------------
``VOH_WALLET_DISABLED=1`` bypasses the ledger and forces a fresh, billed call. It
is off by default, deliberately: a paid call stays protected unless somebody asks
for it in so many words.

Usage example
-------------
>>> import os
>>> os.environ["VOH_WALLET_DISABLED"] = "1"      # keep a doctest from touching a ledger
>>> calls = []
>>> @paid("demo.transcribe")
... def transcribe(*, audio: str) -> str:
...     calls.append(audio)
...     return "text"
>>> transcribe(audio=__file__), len(calls)
('text', 1)
>>> del os.environ["VOH_WALLET_DISABLED"]
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

from vocal_helper._settings import load_settings

__all__ = [
    "audio_fingerprint",
    "paid",
    "wallet_disabled",
    "wallet_path",
]


def wallet_path() -> Path:
    """Resolve the ledger file, environment first.

    Resolution order: ``VOH_WALLET_DB``, then the ``wallet_db`` key of the
    ``paid:`` section of ``settings.yaml``, then ``~/.vocal-helper/wallet.sqlite``.

    **Point an existing ledger here when migrating.** A product that already paid
    for hundreds of hours keeps every one of those results only if vocal-helper
    reads the same file; a fresh default path is a silently empty ledger, and the
    next run re-bills everything.

    Returns
    -------
    pathlib.Path
        The ledger path. Its parent directory is created on first use, not here.

    Examples
    --------
    >>> import os
    >>> os.environ["VOH_WALLET_DB"] = "/tmp/demo-ledger.sqlite"
    >>> wallet_path().as_posix()
    '/tmp/demo-ledger.sqlite'
    >>> del os.environ["VOH_WALLET_DB"]
    >>> wallet_path().name
    'wallet.sqlite'
    """
    explicit = os.environ.get("VOH_WALLET_DB", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    configured = str(load_settings().get("paid", {}).get("wallet_db", "")).strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".vocal-helper" / "wallet.sqlite"


def wallet_disabled() -> bool:
    """Whether ``VOH_WALLET_DISABLED`` asks to bypass memoisation.

    Returns
    -------
    bool
        ``True`` only when the variable is set to something other than ``""``,
        ``0``, ``false`` or ``no``. Anything unset reads as "ledger active", which
        is the safe default: a typo must not start billing.

    Examples
    --------
    >>> import os
    >>> os.environ["VOH_WALLET_DISABLED"] = "1"
    >>> wallet_disabled()
    True
    >>> os.environ["VOH_WALLET_DISABLED"] = "false"
    >>> wallet_disabled()
    False
    >>> del os.environ["VOH_WALLET_DISABLED"]
    >>> wallet_disabled()
    False
    """
    return os.environ.get("VOH_WALLET_DISABLED", "").strip().lower() not in ("", "0", "false", "no")


@functools.lru_cache(maxsize=256)
def _content_sha256(path: str, size: int, mtime: int) -> str:
    """SHA-256 of a file's bytes, memoised per process.

    ``size`` and ``mtime`` are arguments **only** so a rewritten file invalidates
    the in-process memo. They never reach the digest, so they cannot leak into the
    ledger key.

    Parameters
    ----------
    path : str
        File to hash.
    size : int
        Byte size, for memo invalidation only.
    mtime : int
        Modification time, for memo invalidation only.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_fingerprint(path: str) -> str:
    """Content-addressed fingerprint of an audio file, for ledger keying.

    Parameters
    ----------
    path : str
        Path to the audio actually sent to the paid service.

    Returns
    -------
    str
        ``"sha256:<hex>"``. Identical bytes always give an identical string,
        whatever the file is called or when it was written.

    Examples
    --------
    >>> fp = audio_fingerprint(__file__)
    >>> fp.startswith("sha256:") and len(fp) == 71
    True
    >>> fp == audio_fingerprint(__file__)
    True
    """
    stat = os.stat(path)
    return f"sha256:{_content_sha256(path, stat.st_size, int(stat.st_mtime))}"


def _wallet():
    """Build the cross-process wallet over an SQLite ledger.

    Returns
    -------
    wallet_helper.Wallet
        A wallet whose ledger persists at :func:`wallet_path`.
    """
    from wallet_helper import Wallet
    from wallet_helper.sqlite_ledger import SqliteLedger

    path = wallet_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return Wallet(SqliteLedger(str(path)))


def paid(namespace: str, *, key_fields: tuple[str, ...] = ()):
    """Decorate a paid call so the same audio is never billed twice.

    The decorated function **must** take its audio path as the keyword argument
    ``audio``; that is what the key is built from.

    Parameters
    ----------
    namespace : str
        Ledger namespace, e.g. ``"gladia.transcribe"``. Distinct providers and
        distinct operations must not share one.
    key_fields : tuple of str, keyword-only, optional
        Names of the keyword arguments that genuinely change the result and must
        therefore join the audio in the key (typically ``("num_speakers",)``, plus
        ``("diarize",)`` for a transcriber that bundles diarization). Everything
        else is excluded on purpose — see the module docstring.

    Returns
    -------
    callable
        A decorator.

    Notes
    -----
    ``wallet_helper`` is imported **lazily**, on the first genuinely billed call rather
    than when the decorator is applied at import time, so a caller that never
    places a paid call does not pull the dependency in. That matters for a fast
    test lane installed with core extras only.

    Examples
    --------
    >>> import os
    >>> os.environ["VOH_WALLET_DISABLED"] = "1"
    >>> @paid("demo.diarize", key_fields=("num_speakers",))
    ... def diarize(*, audio: str, num_speakers: int = 0) -> int:
    ...     return num_speakers
    >>> diarize(audio=__file__, num_speakers=3)
    3
    >>> del os.environ["VOH_WALLET_DISABLED"]
    """

    def deco(fn):
        # Built on the first real call, then reused: the same one-time cost as eager
        # construction, just deferred past module import.
        state: dict = {}

        def key(*_args, **kwargs):
            # Return the PAYLOAD, not a pre-made key: ``Wallet.call`` already runs
            # ``make_key(namespace, payload)``, so hashing here would double-wrap it.
            payload: dict = {"audio": audio_fingerprint(kwargs["audio"])}
            for field in key_fields:
                if field in kwargs:
                    payload[field] = kwargs[field]
            return payload

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if wallet_disabled():
                return fn(*args, **kwargs)
            if "memoized" not in state:
                from wallet_helper import memoize

                state["memoized"] = memoize(fn, namespace=namespace, key=key, wallet=_wallet())
            return state["memoized"](*args, **kwargs)

        return wrapper

    return deco
