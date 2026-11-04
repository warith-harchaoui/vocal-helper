"""Shared pytest fixtures and isolation hooks.

The package looks for ``settings.yaml`` in the current working
directory and then next to the repo root. On a developer's machine
that file may already exist with a real HF token ; on CI it won't.
Either way, the resolver tests need a *deterministic* environment to
assert the documented order — explicit > env > file.

This conftest replaces :func:`vocal_helper._settings.settings_path`
with an env-only variant for every test : if a test sets
``$VOCAL_HELPER_SETTINGS`` it still resolves through that override,
but the implicit cwd / repo-root lookup is suppressed so an ambient
``settings.yaml`` cannot leak into a test. Ambient ``HF_TOKEN`` is
also unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vocal_helper import _settings


def _env_only_settings_path() -> Path | None:
    """Mirror :func:`settings_path` but skip the cwd / repo-root fallback."""
    # Honour the explicit override env var — that path is set *by the test*
    # itself, so it stays deterministic.
    override = os.environ.get("VOCAL_HELPER_SETTINGS")
    if override:
        p = Path(override).expanduser()
        # A non-existent override is treated as "unset" rather than an error.
        if p.is_file():
            return p
    # No override → return None instead of walking cwd / repo-root, which is
    # the whole point : an ambient settings.yaml must never leak into a test.
    return None


#: The published Nemotron graph's metadata, copied verbatim — except the
#: learned silence embedding, synthetic here : its shape matters, not its values.
NEMOTRON_METADATA = {
    "contract": "sortformer-hr-1",
    "num_speakers": "8",
    "emb_dim": "512",
    "n_mels": "128",
    "subsampling": "8",
    "upsample_factor": "8",
    "spkcache_sil_frames_per_spk": "1",
    "sil_threshold": "0.2",
    "scores_boost_latest": "0.05",
    "pred_score_threshold": "0.25",
    "strong_boost_rate": "0.75",
    "weak_boost_rate": "1.5",
    "min_pos_scores_rate": "0.5",
    "spkcache_len": "264",
    "fifo_len": "40",
    "chunk_len": "340",
    "chunk_right_context": "40",
    "spkcache_update_period": "300",
}


def _proto_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _proto_bytes(field_number: int, payload: bytes) -> bytes:
    return _proto_varint(field_number << 3 | 2) + _proto_varint(len(payload)) + payload


def write_metadata_graph(path: Path, metadata: dict[str, str]) -> Path:
    """A fake ``.onnx`` : a few ``ModelProto`` fields, then ``metadata_props``.

    Field 1 (``ir_version``, varint) and field 7 (``graph``, bytes) are there
    so the reader genuinely has to skip over them.
    """
    body = _proto_varint(1 << 3 | 0) + _proto_varint(8)
    body += _proto_bytes(7, b"\x00" * 1024)
    for key, value in metadata.items():
        entry = _proto_bytes(1, key.encode()) + _proto_bytes(2, value.encode())
        body += _proto_bytes(14, entry)
    path.write_bytes(body)
    return path


@pytest.fixture(autouse=True)
def _no_asr_graph_download(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside integration tests, fetching the ~700 MB ASR graph is an error, not a download."""
    if request.node.get_closest_marker("integration"):
        return
    import vocal_helper.nemotron_asr as nemotron_asr

    def refuse(url: str, target: Path, sha256: str | None) -> None:
        raise FileNotFoundError(f"download refused in unit tests: {url}")

    monkeypatch.setattr(nemotron_asr, "_download", refuse)


@pytest.fixture(scope="session")
def metadata_graph_writer():
    """:func:`write_metadata_graph`, for tests that build their own metadata."""
    return write_metadata_graph


@pytest.fixture(scope="session")
def nemotron_metadata_graph(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The Nemotron graph reduced to its metadata : a readable schedule, no weights."""
    import base64

    import numpy as np

    from vocal_helper.nemotron import META_PREFIX

    silence = np.random.default_rng(0).normal(0, 0.01, 512).astype("<f4")
    metadata = {f"{META_PREFIX}{k}": v for k, v in NEMOTRON_METADATA.items()}
    metadata[f"{META_PREFIX}learnable_sil_emb_f32le_b64"] = base64.b64encode(
        silence.tobytes()
    ).decode()
    return write_metadata_graph(tmp_path_factory.mktemp("nemotron") / "meta.onnx", metadata)


@pytest.fixture(autouse=True)
def _no_bundle_download(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside integration tests, the Nemotron diarizer reads a metadata-only graph.

    Without it, any pipeline construction that asked for the schedule pulled
    the ~1 GB diarization bundle : sixteen minutes of CI, and red whenever the
    served file changed. A graph passed explicitly is left alone.
    """
    if request.node.get_closest_marker("integration"):
        return
    from vocal_helper.nemotron import NemotronOnnxDiar

    fake = request.getfixturevalue("nemotron_metadata_graph")
    real = NemotronOnnxDiar.locate

    def locate(self: NemotronOnnxDiar) -> Path:
        return real(self) if self._graph is not None else fake

    monkeypatch.setattr(NemotronOnnxDiar, "locate", locate)


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ambient HF/settings inputs from every test."""
    # A developer's shell may export HF_TOKEN ; drop it so token-resolution
    # tests see the CI-like "no token" world.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Clear any explicit settings pointer so each test starts from a clean
    # slate and opts back in only when it needs to.
    monkeypatch.delenv("VOCAL_HELPER_SETTINGS", raising=False)
    # Swap the resolver for the env-only variant : suppresses the implicit
    # cwd / repo-root lookup for the duration of the test.
    monkeypatch.setattr(_settings, "settings_path", _env_only_settings_path)
