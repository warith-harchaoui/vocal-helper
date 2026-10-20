"""The public speaker-embedding surface.

These exist because three repositories were importing this module's private names —
notes-helper took _TitaNetEmbedder, _SherpaEmbedder and _resolve_sherpa_models; sidekick
took _SherpaEmbedder. A leading underscore depended on across repositories is not a
dependency, it is a promise nobody made. These lock the promise now made instead.
"""

import pytest

import vocal_helper as vh
from vocal_helper import diar


def test_the_factory_and_the_resolver_are_exported() -> None:
    for name in ("make_embedder", "resolve_sherpa_models", "EMBEDDER_BACKENDS"):
        assert name in vh.__all__, name
        assert hasattr(vh, name), name


def test_nemo_is_the_default_backend() -> None:
    """Picked by studies/diar_embedding_backend.py: 0.354 separability vs 0.201."""
    assert vh.make_embedder().__class__ is diar._TitaNetEmbedder
    assert vh.make_embedder("nemo").__class__ is diar._TitaNetEmbedder


def test_backend_name_is_case_and_space_insensitive() -> None:
    assert vh.make_embedder("  NeMo ").__class__ is diar._TitaNetEmbedder


def test_sherpa_accepts_an_explicit_model_path_without_resolving_anything(monkeypatch) -> None:
    """An explicit path must not trigger resolution — that is the point of passing one."""

    def explode() -> None:
        raise AssertionError("resolution must not run when a path is given")

    monkeypatch.setattr(diar, "_resolve_sherpa_models", explode)
    emb = vh.make_embedder("sherpa", model_path="/tmp/titanet.onnx")
    assert emb.__class__ is diar._SherpaEmbedder


def test_sherpa_without_a_path_resolves_one(monkeypatch) -> None:
    monkeypatch.setattr(diar, "_resolve_sherpa_models", lambda: ("/seg.onnx", "/emb.onnx"))
    assert vh.make_embedder("sherpa").__class__ is diar._SherpaEmbedder


def test_an_unknown_backend_is_named_not_swallowed() -> None:
    with pytest.raises(ValueError, match="unknown embedder backend"):
        vh.make_embedder("maison")


def test_the_two_backends_are_the_whole_choice() -> None:
    """Both emit the same 192-d TitaNet-large vector, so this is an install decision.

    Locking the set matters because the factory is the only supported way in now: a third
    backend appearing without landing here would be unreachable through the public API.
    """
    assert set(diar.EMBEDDER_BACKENDS) == {"nemo", "sherpa"}
    for name in diar.EMBEDDER_BACKENDS:
        assert vh.make_embedder(name, model_path="/tmp/x.onnx") is not None


def test_resolve_sherpa_models_is_the_private_one(monkeypatch) -> None:
    monkeypatch.setattr(diar, "_resolve_sherpa_models", lambda: ("/a.onnx", "/b.onnx"))
    assert diar.resolve_sherpa_models() == ("/a.onnx", "/b.onnx")
