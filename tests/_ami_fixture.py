"""
Self-hosted AMI subset fixture for the offline regression test.

Module summary
--------------
Fetches the small, self-hosted AMI subset (short multi-speaker clips +
clip-relative ground truth) used by ``test_offline_regression.py`` to
guard offline diarization / transcription quality. The subset is hosted
alongside the diarization-engines bundle (both CC BY 4.0 / MIT, no
HuggingFace), so this fixture never touches HF.

Resolution order for the source:

1. ``$VH_AMI_SUBSET`` — a local directory *or* a URL to ``ami-subset.zip``.
2. :data:`DEFAULT_AMI_SUBSET_URL` — the built-in self-hosted default.

A URL is downloaded once and cached under ``$VH_CACHE_DIR`` (default
``~/.cache/vocal-helper``). Contents are verified against the bundled
``manifest.json`` sha256s. On any failure the caller is expected to
``pytest.skip`` — the fixture returns ``None`` rather than raising, so
offline / network-less CI degrades gracefully.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import TypedDict

# Built-in default : the maintainer's self-hosted copy (HF-free).
DEFAULT_AMI_SUBSET_URL = "http://deraison.ai/ami-subset.zip"


class AmiClip(TypedDict):
    """One AMI subset clip with its ground-truth references.

    Attributes
    ----------
    clip_id : str
        Directory / identifier, e.g. ``"IS1008a_w585"``.
    wav : Path
        16 kHz mono WAV of the ~60 s window.
    reference_txt : Path
        Chronological GT word stream (WER reference).
    reference_rttm : Path
        Clip-relative GT speaker turns (DER reference).
    n_speakers : int
        Number of distinct reference speakers in the clip.
    """

    clip_id: str
    wav: Path
    reference_txt: Path
    reference_rttm: Path
    n_speakers: int


def _cache_root() -> Path:
    """Return the cache directory for downloaded fixtures.

    Returns
    -------
    Path
        ``$VH_CACHE_DIR`` if set, else ``~/.cache/vocal-helper``.
    """
    # Honour an explicit override so CI can point at ephemeral storage.
    return Path(os.environ.get("VH_CACHE_DIR", Path.home() / ".cache" / "vocal-helper"))


def _materialize_subset() -> Path | None:
    """Resolve the AMI subset to a local directory, downloading if needed.

    Returns
    -------
    Path or None
        The directory containing ``manifest.json``, or ``None`` when the
        subset cannot be obtained (so the test can skip).
    """
    # Source is an explicit local dir / URL, or the built-in default.
    src = os.environ.get("VH_AMI_SUBSET") or DEFAULT_AMI_SUBSET_URL
    if not src:
        return None

    # Local directory : use it directly if it looks like the subset.
    if not src.startswith(("http://", "https://")):
        p = Path(src).expanduser()
        return p if (p / "manifest.json").exists() else None

    # URL : return the cached extraction if we already have it.
    dest = _cache_root() / "ami-subset"
    hits = list(dest.rglob("manifest.json")) if dest.exists() else []
    if hits:
        return hits[0].parent

    # Otherwise download the zip once and extract it into the cache.
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            urllib.request.urlretrieve(src, tmp.name)
            with zipfile.ZipFile(tmp.name) as archive:
                archive.extractall(dest)
    except Exception:
        # Network down / host unreachable → let the caller skip.
        return None

    hits = list(dest.rglob("manifest.json"))
    return hits[0].parent if hits else None


def _verify(root: Path) -> bool:
    """Check every manifest file exists with the recorded sha256.

    Parameters
    ----------
    root : Path
        Directory containing ``manifest.json``.

    Returns
    -------
    bool
        ``True`` iff all listed clip files match their recorded hash.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    # The subset manifest groups files under each clip entry.
    for clip in manifest.get("clips", []):
        clip_dir = root / clip["clip_id"]
        for fname, meta in clip["files"].items():
            fpath = clip_dir / fname
            if not fpath.exists():
                return False
            # Integrity guard — a truncated download must not pass silently.
            if hashlib.sha256(fpath.read_bytes()).hexdigest() != meta["sha256"]:
                return False
    return True


def load_ami_clips(limit: int | None = None) -> list[AmiClip] | None:
    """Return the AMI subset clips, or ``None`` if unavailable.

    Parameters
    ----------
    limit : int, optional
        Keep only the first ``limit`` clips (deterministic order) to
        bound integration-test runtime. ``None`` returns all clips.

    Returns
    -------
    list of AmiClip or None
        Verified clips, or ``None`` when the subset cannot be fetched or
        fails verification (the test should then skip).

    Examples
    --------
    >>> clips = load_ami_clips(limit=3)   # doctest: +SKIP
    >>> clips and clips[0]["clip_id"]     # doctest: +SKIP
    'ES2011a_w265'
    """
    root = _materialize_subset()
    if root is None or not _verify(root):
        return None

    manifest = json.loads((root / "manifest.json").read_text())
    clips: list[AmiClip] = []
    # Deterministic order (sorted by clip id) so ``limit`` is reproducible.
    for clip in sorted(manifest.get("clips", []), key=lambda c: c["clip_id"]):
        clip_dir = root / clip["clip_id"]
        clips.append(
            AmiClip(
                clip_id=clip["clip_id"],
                wav=clip_dir / "clip.wav",
                reference_txt=clip_dir / "reference.txt",
                reference_rttm=clip_dir / "reference.rttm",
                n_speakers=int(clip["n_speakers"]),
            )
        )
    # Apply the optional cap after sorting for a stable subset.
    return clips[:limit] if limit is not None else clips
