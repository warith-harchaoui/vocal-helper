"""L'hôte de la transcription Nemotron, vérifié sans modèle.

Le graphe ne rend que des jetons par trame. Le front-end, la normalisation, le
découpage en fenêtres et la reconstruction des mots vivent autour de lui, en
numpy pur : un graphe réduit à ses métadonnées et une fausse session suffisent.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np
import pytest

from vocal_helper.nemotron_asr import (
    CONTRACT,
    GRAPH_ENV,
    META_PREFIX,
    AsrSettings,
    CtcToken,
    NemotronOnnxAsr,
    greedy_ctc,
    log_mel,
    normalize_per_feature,
    tokens_to_words,
)

VOCAB = ["<unk>", "<spk:0>", "<ev:laughter>", "▁bon", "jour", "▁le", "▁monde"]
BLANK = len(VOCAB)


def _metadata(**override: str) -> dict[str, str]:
    rng = np.random.default_rng(0)
    fb = np.abs(rng.normal(0, 1, (128, 257))).astype("<f4")
    window = np.hanning(400).astype("<f4")
    vocab = json.dumps(VOCAB, ensure_ascii=False)
    meta = {
        "contract": CONTRACT,
        "sample_rate": "16000",
        "n_fft": "512",
        "hop_length": "160",
        "win_length": "400",
        "preemphasis": "0.97",
        "log_guard": repr(2.0**-24),
        "n_mels": "128",
        "normalize": "per_feature",
        "stack": "8",
        "frame_s": "0.04",
        "chunk_s": "30.0",
        "blank_id": str(BLANK),
        "vocab": vocab,
        "mel_fb": base64.b64encode(fb.tobytes()).decode(),
        "mel_window": base64.b64encode(window.tobytes()).decode(),
    }
    meta.update(override)
    return {META_PREFIX + k: v for k, v in meta.items()}


@pytest.fixture
def asr_graph(tmp_path: Path, metadata_graph_writer) -> Path:
    return metadata_graph_writer(tmp_path / "asr.onnx", _metadata())


def test_settings_read_from_metadata_alone(asr_graph: Path) -> None:
    s = NemotronOnnxAsr(asr_graph).settings
    assert s.fb.shape == (128, 257)
    assert s.window.shape == (400,)
    assert s.vocab == tuple(VOCAB)
    assert s.blank_id == BLANK
    assert s.chunk_s == 30.0


def test_unknown_contract_is_refused(tmp_path: Path, metadata_graph_writer) -> None:
    graph = metadata_graph_writer(tmp_path / "x.onnx", _metadata(contract="autre-1"))
    with pytest.raises(ValueError, match="contrat"):
        _ = NemotronOnnxAsr(graph).settings


def test_without_a_graph_the_default_is_fetched_once_into_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vocal_helper.nemotron_asr as nemotron_asr

    monkeypatch.delenv(GRAPH_ENV, raising=False)
    monkeypatch.delenv(nemotron_asr.URL_ENV, raising=False)
    monkeypatch.setenv("VH_CACHE_DIR", str(tmp_path))
    fetched: list[tuple[str, str | None]] = []

    def fake(url: str, target: Path, sha256: str | None) -> None:
        fetched.append((url, sha256))
        target.write_bytes(b"x")

    monkeypatch.setattr(nemotron_asr, "_download", fake)
    path = NemotronOnnxAsr(precision="fp32").locate()
    assert path == tmp_path / "nemotron-onnx" / "nemotron-asr-ctc.fp32.onnx"
    assert [u.rsplit("/", 1)[1] for u, _ in fetched] == [
        "nemotron-asr-ctc.fp32.onnx",
        "nemotron-asr-ctc.fp32.onnx.data",
    ]
    assert all(u.startswith("https://deraison.ai/nemotron-onnx/") and d for u, d in fetched)
    NemotronOnnxAsr(precision="fp32").locate()
    assert len(fetched) == 2  # le cache sert la seconde fois


def test_a_mirror_is_not_held_to_the_default_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vocal_helper.nemotron_asr as nemotron_asr

    monkeypatch.delenv(GRAPH_ENV, raising=False)
    monkeypatch.setenv(nemotron_asr.URL_ENV, "https://mirror.example/onnx/")
    monkeypatch.setenv("VH_CACHE_DIR", str(tmp_path))
    fetched: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        nemotron_asr,
        "_download",
        lambda url, target, sha256: fetched.append((url, sha256)) or target.write_bytes(b"x"),
    )
    NemotronOnnxAsr().locate()
    assert fetched == [("https://mirror.example/onnx/nemotron-asr-ctc.int8.onnx", None)]


def test_unknown_precision_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GRAPH_ENV, raising=False)
    with pytest.raises(ValueError, match="précision"):
        NemotronOnnxAsr(precision="fp8").locate()


def test_graph_from_env(asr_graph: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GRAPH_ENV, os.fspath(asr_graph))
    assert NemotronOnnxAsr().locate() == asr_graph


def test_log_mel_matches_nemo_frame_count_and_zeroes_the_tail(asr_graph: Path) -> None:
    s = NemotronOnnxAsr(asr_graph).settings
    audio = np.random.default_rng(1).normal(0, 0.1, 16000 + 37).astype(np.float32)
    mel, valid = log_mel(audio, s)
    assert valid == (16000 + 37) // 160
    assert mel.shape == (128, 1 + (16037 + 512 - 512) // 160)
    assert np.all(mel[:, valid:] == 0.0)
    assert np.all(np.isfinite(mel))


def test_normalize_ignores_padding() -> None:
    mel = np.zeros((3, 10), np.float32)
    mel[:, :6] = np.arange(18, dtype=np.float32).reshape(3, 6)
    mel[:, 6:] = 1e6
    out = normalize_per_feature(mel, 6)
    assert np.allclose(out[:, :6].mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(out[:, :6].std(axis=1, ddof=1), 1.0, atol=1e-4)
    assert np.all(out[:, 6:] == 0.0)


def test_greedy_ctc_keeps_repeats_split_by_blank() -> None:
    ids = np.array([BLANK, 3, 3, BLANK, 3, 4, 4])
    toks = greedy_ctc(ids, np.zeros(7), BLANK)
    assert [(t.id, t.first, t.last) for t in toks] == [(3, 1, 2), (3, 4, 4), (4, 5, 6)]


def test_words_tags_and_confidence() -> None:
    toks = [
        CtcToken(1, 0, 0, 0.0),
        CtcToken(3, 2, 3, np.log(0.5)),
        CtcToken(4, 4, 4, np.log(0.5)),
        CtcToken(2, 6, 6, 0.0),
        CtcToken(5, 8, 8, 0.0),
        CtcToken(6, 9, 11, 0.0),
    ]
    words, tags = tokens_to_words(toks, tuple(VOCAB), 0.04)
    assert [w.text for w in words] == ["bonjour", "le", "monde"]
    assert words[0].start_s == 0.08 and words[0].end_s == 0.2
    assert words[0].confidence == pytest.approx(0.5)
    assert tags == [(0.0, "<spk:0>"), (0.24, "<ev:laughter>")]


class _FakeSession:
    """Rend « bonjour » dans chaque fenêtre, et note les longueurs reçues."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def run(self, _outputs, feeds):
        mel, enc_len = feeds["mel"], feeds["enc_len"]
        assert mel.shape[2] % 8 == 0 and mel.dtype == np.float32
        frames = mel.shape[2] // 4
        self.calls.append((mel.shape[2], int(enc_len[0])))
        ids = np.full((1, frames), BLANK, np.int32)
        ids[0, 1:3], ids[0, 3] = 3, 4
        best = np.zeros((1, frames), np.float32)
        blank = np.where(ids == BLANK, 0.0, -5.0).astype(np.float32)
        return ids, best, blank


def test_long_audio_is_cut_in_30s_windows(asr_graph: Path) -> None:
    asr = NemotronOnnxAsr(asr_graph)
    fake = _FakeSession()
    asr._session = fake
    audio = np.random.default_rng(2).normal(0, 0.1, 16000 * 65 + 123).astype(np.float32)
    out = asr.transcribe(audio, 16000)
    assert [c[0] for c in fake.calls] == [3000, 3000, 504]
    assert all(mel == enc * 8 for mel, enc in fake.calls)
    assert [w.text for w in out.words] == ["bonjour"] * 3
    assert [w.start_s for w in out.words] == [0.04, 30.04, 60.04]
    assert out.speech.shape == (-(-(len(audio) // 160) // 4),)
    assert out.speech.max() > 0.99


def test_wrong_rate_is_refused(asr_graph: Path) -> None:
    with pytest.raises(ValueError, match="16000"):
        NemotronOnnxAsr(asr_graph).frames(np.zeros(800, np.float32), 8000)


def test_settings_dataclass_defaults() -> None:
    s = AsrSettings(np.ones((128, 257), np.float32), np.ones(400, np.float32), (), 0)
    assert s.stack == 8 and s.frame_s == 0.04


@pytest.mark.integration
def test_real_graph_transcribes_speech() -> None:
    graph = os.environ.get(GRAPH_ENV)
    if not graph:
        pytest.skip(f"${GRAPH_ENV} non défini")
    import soundfile as sf

    speech = Path.home() / "pasdebonneoudemauvaisesituation/data/speech.wav"
    if not speech.exists():
        pytest.skip("extrait de parole absent")
    audio, sr = sf.read(speech, dtype="float32")
    out = NemotronOnnxAsr(graph).transcribe(audio if audio.ndim == 1 else audio.mean(1), sr)
    assert len(out.words) > 3


def test_streaming_commits_every_frame_once_within_its_delay(asr_graph: Path) -> None:
    asr = NemotronOnnxAsr(asr_graph)
    asr._session = _FakeSession()
    audio = np.random.default_rng(3).normal(0, 0.1, 16000 * 11 + 300).astype(np.float32)
    ids, best, blank, delay = asr.streaming_frames(
        audio, 16000, hop_s=2.0, left_s=4.0, right_s=0.48
    )
    assert len(ids) == len(best) == len(blank) == len(delay) == -(-len(audio) // 640)
    # Retard nominal entre l'anticipation et l'anticipation plus un pas ; la
    # fin du flux est vidée sans anticipation, donc plus tôt.
    nominal = delay[: -int(2.48 / 0.04)]
    assert nominal.min() >= 0.48 - 1e-6
    assert nominal.max() <= 2.48 + 0.5


def test_streaming_window_is_bounded_by_the_trained_length(asr_graph: Path) -> None:
    with pytest.raises(ValueError, match="apprises"):
        NemotronOnnxAsr(asr_graph).streaming_frames(
            np.zeros(16000, np.float32), 16000, hop_s=4.0, left_s=28.0
        )


def _segment(t0: float, seconds: float, speaker: str) -> dict:
    return {
        "t0": t0,
        "t1": t0 + seconds,
        "sample_rate": 16000,
        "speaker": speaker,
        "pcm": np.zeros(int(seconds * 16000), np.float32),
    }


def test_stage_gives_each_word_back_to_its_segment(asr_graph: Path) -> None:
    from vocal_helper.nemotron_asr import NemotronAsrStage, Transcript, Word

    stage = NemotronAsrStage(asr_graph, batch=True)
    # Deux segments de 1 s séparés de 0,1 s : le second commence à 1,1 s local.
    stage._asr.transcribe = lambda audio, sr: Transcript(
        words=[Word("salut", 0.2, 0.5, 0.9), Word("bonjour", 1.3, 1.8, 0.8)]
    )
    utts = stage.transcribe_chunk([_segment(10.0, 1.0, "S0"), _segment(42.0, 1.0, "S1")])
    assert [u["text"] for u in utts] == ["salut", "bonjour"]
    assert [u["speaker"] for u in utts] == ["S0", "S1"]
    assert utts[1]["words"][0][0] == pytest.approx(42.2)
    assert utts[0]["language"] is None


@pytest.mark.parametrize("batch", [True, False])
def test_stage_run_keeps_one_utterance_per_segment(asr_graph: Path, batch: bool) -> None:
    import asyncio

    from vocal_helper.nemotron_asr import NemotronAsrStage, Transcript

    stage = NemotronAsrStage(asr_graph, batch=batch)
    calls: list[int] = []
    stage._asr.transcribe = lambda audio, sr: calls.append(len(audio)) or Transcript()

    async def drive() -> list:
        inbox, outbox = asyncio.Queue(), asyncio.Queue()
        for i in range(3):
            await inbox.put(_segment(float(i), 1.0, f"S{i}"))
        await inbox.put(None)
        await stage.run(inbox, outbox)
        out = []
        while (item := await outbox.get()) is not None:
            out.append(item)
        return out

    utts = asyncio.run(drive())
    assert [u["speaker"] for u in utts] == ["S0", "S1", "S2"]
    assert len(calls) == (1 if batch else 3)


def test_pipeline_picks_the_engine_by_name(asr_graph: Path) -> None:
    from vocal_helper.nemotron_asr import NemotronAsrStage
    from vocal_helper.pipeline import _asr_stage

    stage = _asr_stage({"engine": "nemotron", "graph": asr_graph, "warmup": True, "batch": True})
    assert isinstance(stage, NemotronAsrStage) and stage.batch
    with pytest.raises(ValueError, match="inconnu"):
        _asr_stage({"engine": "gladia"})


@pytest.mark.parametrize(
    ("config", "offline", "expected"),
    [
        ({}, True, "nemotron"),
        (
            {"language": "auto", "model": "large-v3-turbo-q5_0", "initial_prompt": ""},
            True,
            "nemotron",
        ),
        ({}, False, "nemotron"),
        ({"language": "fr", "warmup": True}, False, "whisper"),
        ({"language": "fr"}, True, "whisper"),
        ({"initial_prompt": "réunion budgétaire"}, True, "whisper"),
        ({"model": "small"}, True, "whisper"),
        ({"engine": "nemotron"}, False, "nemotron"),
        ({"engine": "whisper"}, True, "whisper"),
    ],
)
def test_auto_engine_is_the_measured_best(config: dict, offline: bool, expected: str) -> None:
    """``auto`` : Nemotron quand la langue est à découvrir, hors ligne comme en direct ; whisper sinon."""
    from vocal_helper.pipeline import choose_asr_engine

    assert choose_asr_engine(config, offline=offline) == expected
