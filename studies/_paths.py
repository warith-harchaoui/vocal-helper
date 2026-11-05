"""Where the studies find their corpora, weights and logs.

No study names a machine. Every location comes from an environment variable;
when the variable is unset the default is a folder of this checkout (all
git-ignored), so a fresh clone runs the same command with the data dropped in
place. A location with no sensible default raises a clear error instead.

==================  =========================================================
Variable            What it points at (default)
==================  =========================================================
VH_STUDY_DATA_ROOT  The bench data root: ``ami/dev-slice/``,
                    ``bagarre-rich/``, ``common-voice-subset/``,
                    ``speech.wav`` (``data/``)
VH_AMI_ROOT         The AMI dev-slice, one folder per meeting holding
                    ``mix.wav`` + ``words.rttm``
                    (``$VH_STUDY_DATA_ROOT/ami/dev-slice``)
VH_TV_CORPUS        The French TV episodes built by the paid transcription
                    path (``$VH_STUDY_DATA_ROOT/francetv-corpus/output``)
VH_RUN_LOGS         Where a long sweep tees its log (``run-logs/``)
VH_MODELS_DIR       Local whisper weights, ggml and MLX (``models/``)
VH_BENCH_SRC        The diarization bench's source tree, imported by
                    ``diar_embedding_backend.py`` only (no default)
VH_NEMOTRON_STUDY   The Nemotron export folder holding ``lite/`` and
                    ``onnx/``, named in the usage lines (no default)
==================  =========================================================
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


def data_root() -> Path:
    """The bench data root (``$VH_STUDY_DATA_ROOT``, else ``data/``)."""
    return _env_path("VH_STUDY_DATA_ROOT", REPO / "data")


def ami_root() -> Path:
    """The AMI dev-slice (``$VH_AMI_ROOT``, else ``<data root>/ami/dev-slice``)."""
    return _env_path("VH_AMI_ROOT", data_root() / "ami" / "dev-slice")


def tv_corpus() -> Path:
    """The French TV episodes (``$VH_TV_CORPUS``)."""
    return _env_path("VH_TV_CORPUS", data_root() / "francetv-corpus" / "output")


def models_dir() -> Path:
    """Local whisper weights (``$VH_MODELS_DIR``, else ``models/``)."""
    return _env_path("VH_MODELS_DIR", REPO / "models")


def run_log(name: str) -> Path:
    """A sweep's log file under ``$VH_RUN_LOGS`` (else ``run-logs/``)."""
    return _env_path("VH_RUN_LOGS", REPO / "run-logs") / name


def required(var: str, what: str) -> Path:
    """A location with no default: the variable must be set.

    Raises
    ------
    SystemExit
        When ``var`` is unset, naming the variable and what it points at.
    """
    value = os.environ.get(var)
    if not value:
        raise SystemExit(f"{var} is not set: point it at {what}.")
    return Path(value).expanduser()
