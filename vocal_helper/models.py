"""
vocal_helper.models
====================

Model registry + on-demand downloader for the diarization / speaker-embedding
ONNX weights vocal-helper's torch-free ``sherpa`` path needs.

Sourcing policy (sovereign, HuggingFace-free at runtime)
--------------------------------------------------------
Weights are fetched **on first use** and cached on disk. The registry resolves
each model from the **shared ai-helpers mirror** — the same host and cache the
rest of the suite uses (``video-helper``'s face stack, etc.), so a weight
downloaded once by any helper is reused by all:

- ``AI_HELPERS_MODEL_BASE_URL`` (default
  ``https://harchaoui.org/warith/ai-helpers/models/``) — the user's own
  infrastructure. Never HuggingFace, no token, no gated repo.

If the mirror (and any HuggingFace-free upstream fallback) fails, the caller
gets ``None`` and is expected to degrade gracefully — e.g. ``diar._SherpaEmbedder``
falls back to raising a clear "provide a model_path" error, and the existing
env-override / diarization-engines-bundle resolution keeps working untouched.

Everything is logged through ``os_helper`` (``osh.info/warning``); files land
under ``~/.cache/ai-helpers/models/`` (override with ``VOCAL_HELPER_MODEL_DIR``)
— the same shared cache directory the sibling helpers use.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import os_helper as osh

# Default mirror on the user's own infrastructure, shared across every
# ai-helper. Runtime downloads prefer this so there is no third-party (and
# specifically no HuggingFace) dependency. Override for testing / a private
# mirror. The env var name is shared with the sibling helpers on purpose.
DEFAULT_BASE_URL = "https://harchaoui.org/warith/ai-helpers/models/"

# Licenses that clear the default download gate — all permit commercial use.
# CC-BY-4.0 / CC-BY-SA-4.0 cover the NVIDIA NeMo speaker exports (attribution
# only); the pyannote segmentation ONNX exports are MIT.
_PERMISSIVE_LICENSES = frozenset({"Apache-2.0", "MIT", "BSD-3-Clause", "CC-BY-4.0", "CC-BY-SA-4.0"})


def _base_url() -> str:
    """Return the mirror base URL (env override, trailing slash guaranteed)."""
    return os.environ.get("AI_HELPERS_MODEL_BASE_URL", DEFAULT_BASE_URL).rstrip("/") + "/"


def model_dir() -> str:
    """Return (creating if needed) the shared local model cache directory.

    Defaults to ``~/.cache/ai-helpers/models`` — the same location the other
    ai-helpers use, so a weight fetched by one is reused by all. Override with
    ``VOCAL_HELPER_MODEL_DIR``.

    Returns
    -------
    str
        Absolute path to the (now existing) cache directory.
    """
    d = os.environ.get("VOCAL_HELPER_MODEL_DIR") or os.path.expanduser("~/.cache/ai-helpers/models")
    osh.make_directory(d)
    return d


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable weight file.

    Parameters
    ----------
    name : str
        Registry key (also the config-facing identifier).
    filename : str
        On-disk basename under the cache dir and the path segment on the mirror.
    sha256 : str
        Expected hex digest, or ``""`` to skip integrity checking (used until
        the mirror is seeded with pinned digests).
    upstreams : list[str]
        HuggingFace-free fallback URLs tried, in order, only if the mirror misses.
    license : str
        SPDX-ish tag; a model whose license is not clearly permissive is gated
        by the caller (``allow_noncommercial``).
    """

    name: str
    filename: str
    sha256: str = ""
    upstreams: list[str] = field(default_factory=list)
    license: str = "unknown"


# The ONNX weights the torch-free ``sherpa`` diarization path uses. Every one
# is hosted on the shared mirror (verified HTTP 200). Filenames match exactly
# the names ``diar._resolve_sherpa_models`` already looks for in the bundle, so
# the mirror is a drop-in third resolution tier after env / bundle.
REGISTRY: dict[str, ModelSpec] = {
    # Speaker embedding — NVIDIA TitaNet-large (the study-selected best embedder,
    # FR+EN). Default for both the online and offline ``sherpa`` backends.
    "sherpa-titanet-large": ModelSpec(
        name="sherpa-titanet-large",
        filename="nemo_en_titanet_large.onnx",
        license="CC-BY-4.0",
    ),
    # The fast twin — TitaNet-small, the embedding fallback in the resolver.
    "sherpa-titanet-small": ModelSpec(
        name="sherpa-titanet-small",
        filename="nemo_en_titanet_small.onnx",
        license="CC-BY-4.0",
    ),
    # Speaker segmentation — pyannote community-1, our sovereign ONNX export.
    # The default segmentation for the offline ``sherpa`` pipeline.
    "community1-segmentation": ModelSpec(
        name="community1-segmentation",
        filename="community1-segmentation.onnx",
        license="MIT",
    ),
    # Segmentation alternative — the pyannote segmentation-3.0 ONNX export.
    "pyannote-segmentation-3": ModelSpec(
        name="pyannote-segmentation-3",
        filename="pyannote_segmentation_3.onnx",
        license="MIT",
    ),
}


def _verify(path: str, sha256: str) -> bool:
    """Return True if ``path`` matches ``sha256`` (or no digest was pinned)."""
    if not sha256:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    ok = h.hexdigest() == sha256
    if not ok:
        osh.warning(f"models: checksum mismatch for {os.path.basename(path)}")
    return ok


def ensure_model(name: str, *, allow_noncommercial: bool = False) -> str | None:
    """Resolve a model to a local path, downloading + caching on first use.

    Parameters
    ----------
    name : str
        A key in :data:`REGISTRY`.
    allow_noncommercial : bool, optional
        Gate for research / non-commercial weights. A model whose license is not
        clearly permissive is refused unless this is set. Default ``False``.

    Returns
    -------
    str or None
        Local filesystem path to the ready weight, or ``None`` if the model is
        gated-off or could not be fetched from any source (caller degrades).
    """
    spec = REGISTRY.get(name)
    if spec is None:
        osh.warning(f"models: unknown model {name!r}")
        return None

    if spec.license not in _PERMISSIVE_LICENSES and not allow_noncommercial:
        osh.warning(
            f"models: {name!r} is '{spec.license}' — pass allow_noncommercial=True to use it"
        )
        return None

    dest = osh.join(model_dir(), spec.filename)
    if osh.file_exists(dest) and _verify(dest, spec.sha256):
        return dest

    # Prefer the shared mirror, then any HuggingFace-free upstreams.
    sources = [_base_url() + spec.filename, *spec.upstreams]
    for url in sources:
        try:
            osh.info(f"models: fetching {name} from {url}")
            osh.download_file(url, dest)
            if _verify(dest, spec.sha256):
                return dest
        except Exception as exc:  # noqa: BLE001 — try the next source, never crash the pipeline
            osh.warning(f"models: source failed ({url}): {exc}")
            continue

    osh.warning(
        f"models: could not obtain {name!r} from any source — the caller must degrade gracefully"
    )
    return None
