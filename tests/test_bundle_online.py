"""Ce que le serveur sert réellement, contre ce que le défaut exige.

Réseau, donc marqué ``integration`` : hors CI et hors du crochet pre-push,
à lancer à la main (``pytest -m integration tests/test_bundle_online.py``)
après chaque publication de poids. C'est la garde qui a manqué quand le
ZIP servi a cessé d'être celui que le code épingle.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pytest

pytestmark = pytest.mark.integration

#: Taille du graphe dans ``manifest.json`` du bundle qui le porte.
GRAPH_BYTES = 400_128_655


def _head(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200, url
        return int(response.headers["Content-Length"])


class _RangeReader(io.RawIOBase):
    """Un fichier distant lu par plages, pour lister un ZIP sans le télécharger."""

    def __init__(self, url: str) -> None:
        self.url, self.pos, self.size = url, 0, _head(url)

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def]
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def test_the_single_nemotron_graph_is_served_whole():
    """Le défaut (Nemotron) doit être atteignable sur une installation neuve."""
    from vocal_helper.nemotron import DEFAULT_BASE_URL, GRAPH_NAME

    assert _head(f"{DEFAULT_BASE_URL}/{GRAPH_NAME}") == GRAPH_BYTES


def test_the_default_bundle_carries_what_the_offline_backends_read():
    from vocal_helper.diar import DEFAULT_DIARIZATION_ENGINES_URL

    assert DEFAULT_DIARIZATION_ENGINES_URL is not None
    archive = zipfile.ZipFile(io.BufferedReader(_RangeReader(DEFAULT_DIARIZATION_ENGINES_URL)))
    names = set(archive.namelist())
    for required in (
        "diarization-engines/manifest.json",
        "diarization-engines/pyannote-3.1/pyannote_diarization_config.yaml",
        "diarization-engines/pyannote-embedding/pytorch_model.bin",
        "diarization-engines/sherpa/nemo_en_titanet_large.onnx",
    ):
        assert required in names, required


def test_the_served_bundle_matches_the_pinned_digest(tmp_path):
    """Le téléchargement complet (≈ 800 Mo) : une épingle fausse casse toute installation neuve."""
    import os_helper as osh

    from vocal_helper.diar import (
        DEFAULT_DIARIZATION_ENGINES_SHA256,
        DEFAULT_DIARIZATION_ENGINES_URL,
    )

    assert DEFAULT_DIARIZATION_ENGINES_URL is not None
    osh.download_file(
        DEFAULT_DIARIZATION_ENGINES_URL,
        str(tmp_path / "bundle.zip"),
        sha256=DEFAULT_DIARIZATION_ENGINES_SHA256,
    )
