"""Ce que la chaîne en direct choisit par défaut, et ce qu'un choix explicite change.

Le graphe Nemotron est remplacé par sa seule en-tête de métadonnées dans les
tests unitaires (``conftest._no_bundle_download``) : il est donc toujours « là »,
ce qui rend déterministe la question de savoir s'il est choisi.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


async def _source():
    yield {"t0": 0.0, "sample_rate": 16000, "pcm": np.zeros(320, dtype=np.float32)}


def test_the_pipeline_latency_default_is_the_engines_measured_default():
    """En salle réverbérée, 1,04 s fait 0,252 contre 0,350 à 6,4 s : la chaîne
    ne doit pas défaire en douce le défaut que le moteur a pris sur mesure."""
    import vocal_helper as voh
    from vocal_helper.nemotron import DEFAULT_LATENCY_S, NemotronOnnxDiar

    assert voh.PipelineConfig().diar_latency_s == DEFAULT_LATENCY_S
    pipeline = voh.Pipeline(source=_source, config=voh.PipelineConfig(llm=None))
    assert isinstance(pipeline._diar, NemotronOnnxDiar)
    assert pipeline._diar.schedule.latency_s == pytest.approx(DEFAULT_LATENCY_S)


def test_none_still_reaches_the_longest_context():
    import vocal_helper as voh

    longest = voh.Pipeline(source=_source, config=voh.PipelineConfig(llm=None, diar_latency_s=None))
    assert longest._diar.schedule.latency_s == pytest.approx(6.4)


@pytest.mark.parametrize("backend", ["pyannote", "nemo", "sherpa"])
def test_an_explicit_backend_is_not_replaced_by_nemotron(backend):
    """Le README passe ``diar={"backend": "pyannote"}`` : c'est un choix, il gagne."""
    import vocal_helper as voh
    from vocal_helper.growing import GrowingBufferDiarStage
    from vocal_helper.nemotron import NemotronOnnxDiar

    pipeline = voh.Pipeline(
        source=_source, config=voh.PipelineConfig(llm=None, diar={"backend": backend})
    )
    assert not isinstance(pipeline._diar, NemotronOnnxDiar)
    assert isinstance(pipeline._diar, GrowingBufferDiarStage)
    # La passe différée suit le même choix au lieu de résoudre son propre "auto"
    # vers Nemotron.
    assert pipeline._diar.offline.backend == backend


def test_a_paid_backend_is_not_repeated_by_the_growing_buffer():
    import vocal_helper as voh

    pipeline = voh.Pipeline(
        source=_source, config=voh.PipelineConfig(llm=None, diar={"backend": "pyannoteai"})
    )
    assert pipeline._diar.offline.backend != "pyannoteai"


def test_naming_nemotron_or_auto_keeps_nemotron():
    import vocal_helper as voh
    from vocal_helper.nemotron import NemotronOnnxDiar

    for diar in ({}, {"backend": "auto"}):
        pipeline = voh.Pipeline(source=_source, config=voh.PipelineConfig(llm=None, diar=diar))
        assert isinstance(pipeline._diar, NemotronOnnxDiar)


# --------------------------------------------------------------------------- #
#  Le graphe seul, quand le bundle ne le porte pas
# --------------------------------------------------------------------------- #


@pytest.fixture
def bundle_without_nemotron(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    import vocal_helper.diar as diar

    monkeypatch.setattr(diar, "resolve_diarization_engines", lambda: bundle)
    monkeypatch.setenv("VH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("VH_NEMOTRON_DIAR_URL", raising=False)
    return bundle


def test_a_bundle_without_the_graph_falls_back_to_the_single_file(
    bundle_without_nemotron, monkeypatch
):
    from vocal_helper import nemotron

    calls = []

    def fake(url: str, target: Path, sha256: str | None) -> None:
        calls.append((url, sha256))
        target.write_bytes(b"graph")

    monkeypatch.setattr(nemotron, "_download", fake)
    path = nemotron.resolve_graph()
    assert calls == [
        (
            "https://deraison.ai/nemotron-onnx/nemotron-3-diarization/nemotron-3-diarization.onnx",
            nemotron.GRAPH_SHA256,
        )
    ]
    assert path.parent.name == "nemotron-onnx"  # le cache du graphe ASR
    # Une seconde fois, le cache suffit.
    assert nemotron.resolve_graph() == path
    assert len(calls) == 1


def test_a_mirror_is_not_held_to_the_default_digest(bundle_without_nemotron, monkeypatch):
    from vocal_helper import nemotron

    calls = []
    monkeypatch.setenv("VH_NEMOTRON_DIAR_URL", "https://mirror.example/x/")
    monkeypatch.setattr(
        nemotron, "_download", lambda u, t, s: (calls.append((u, s)), t.write_bytes(b"g"))
    )
    nemotron.resolve_graph()
    assert calls == [("https://mirror.example/x/nemotron-3-diarization.onnx", None)]


def test_the_bundle_wins_when_it_carries_the_graph(bundle_without_nemotron, monkeypatch):
    from vocal_helper import nemotron

    graph = bundle_without_nemotron / "nemotron3" / "nemotron-3-diarization.onnx"
    graph.parent.mkdir()
    graph.write_bytes(b"g")
    monkeypatch.setattr(nemotron, "_download", lambda *a: pytest.fail("no download"))
    assert nemotron.resolve_graph() == graph


def test_an_unreachable_graph_reads_as_absent(bundle_without_nemotron, monkeypatch):
    """La chaîne dégrade vers la fenêtre croissante sur FileNotFoundError."""
    from vocal_helper import nemotron

    def boom(*_a):
        raise OSError("404")

    monkeypatch.setattr(nemotron, "_download", boom)
    with pytest.raises(FileNotFoundError, match="404"):
        nemotron.resolve_graph()
