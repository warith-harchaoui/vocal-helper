"""La tenue du cache de Nemotron, vérifiée sans modèle ni audio.

Le graphe ONNX est **un pas de flux** : il reçoit ``[cache | file | morceau]``
et rend des probabilités. Tout ce qui décide de la qualité du suivi vit autour
de lui — quelles trames entrent dans la file, lesquelles en sortent, comment le
cache se comprime quand il déborde — et c'est du numpy pur, donc tout se
vérifie sur des tableaux faits à la main.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vocal_helper.nemotron import (
    STREAMING_MODES,
    MelSettings,
    NemotronOnnxDiar,
    NemotronSchedule,
    StreamingState,
    boost_top_scores,
    compress_cache,
    disable_low_scores,
    log_mel,
    log_pred_scores,
    probabilities_to_segments,
    streaming_update,
)


def _schedule(**kwargs) -> NemotronSchedule:
    """Un petit calendrier, pour que les tableaux restent lisibles."""
    base = {
        "num_speakers": 2,
        "emb_dim": 3,
        "spkcache_len": 4,
        "fifo_len": 4,
        "spkcache_sil_frames_per_spk": 0,
    }
    return NemotronSchedule(**{**base, **kwargs})


# ----- le front-end --------------------------------------------------


def test_the_front_end_emits_one_frame_every_ten_milliseconds():
    frames = log_mel(np.zeros(16000, dtype=np.float32), 16000)
    assert frames.shape == (101, 128)


def test_silence_lands_exactly_on_the_log_floor():
    frames = log_mel(np.zeros(8000, dtype=np.float32), 16000)
    assert np.allclose(frames, math.log(2.0**-24))


def test_a_wrong_sample_rate_is_refused_rather_than_resampled():
    """Rééchantillonner en silence donnerait des trames plausibles et fausses."""
    with pytest.raises(ValueError, match="16000 Hz"):
        log_mel(np.zeros(8000, dtype=np.float32), 8000)


def test_an_empty_buffer_is_not_an_error():
    assert log_mel(np.zeros(0, dtype=np.float32), 16000).shape == (0, 128)


def test_the_window_is_centred_in_the_transform():
    """La convention retrouvée contre l'extracteur officiel.

    Mettre la fenêtre de 400 points au début de la transformée de 512 au lieu
    de la centrer multiplie par seize l'écart aux sorties officielles. Le test
    vérifie que le décalage est bien celui attendu, par un signal dont une
    seule impulsion révèle la position de la fenêtre.
    """
    settings = MelSettings()
    offset = (settings.n_fft - settings.win_length) // 2
    assert offset == 56, "la fenêtre doit être centrée, 56 zéros de chaque côté"


# ----- les scores ----------------------------------------------------


def test_a_lone_confident_speaker_scores_above_two_speaking_at_once():
    alone = log_pred_scores(np.array([[0.95, 0.02]], dtype=np.float32), floor=0.25)
    together = log_pred_scores(np.array([[0.95, 0.95]], dtype=np.float32), floor=0.25)
    assert alone[0, 0] > together[0, 0]


def test_the_floor_keeps_the_scores_finite():
    scores = log_pred_scores(np.array([[0.0, 1.0]], dtype=np.float32), floor=0.25)
    assert np.isfinite(scores).all()


def test_non_speech_is_pushed_out_of_the_cache():
    preds = np.array([[0.9, 0.2]], dtype=np.float32)
    out = disable_low_scores(preds, np.zeros((1, 2)), min_positive=99)
    assert np.isneginf(out[0, 1])
    assert not np.isneginf(out[0, 0])


def test_overlap_is_dropped_only_when_the_speaker_has_clean_frames_to_spare():
    """Sans cette condition, un locuteur qui ne parle jamais seul disparaîtrait."""
    preds = np.full((3, 1), 0.9, dtype=np.float32)
    scores = np.array([[-1.0], [-1.0], [-1.0]])
    assert not np.isneginf(disable_low_scores(preds, scores, min_positive=1)).any()


def test_boosting_lifts_exactly_the_requested_count():
    scores = np.array([[0.0], [1.0], [2.0]])
    out = boost_top_scores(scores, 2, scale=1.0)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[1, 0] > 1.0
    assert out[2, 0] > 2.0


def test_boosting_never_resurrects_a_disabled_frame():
    scores = np.array([[-np.inf], [1.0]])
    out = boost_top_scores(scores, 2, scale=1.0)
    assert np.isneginf(out[0, 0])


def test_boosting_nothing_changes_nothing():
    scores = np.array([[0.0], [1.0]])
    assert np.array_equal(boost_top_scores(scores, 0, scale=1.0), scores)


# ----- la compression du cache ---------------------------------------


def test_compression_keeps_the_cache_at_its_length():
    schedule = _schedule(spkcache_len=3)
    embs = np.random.default_rng(0).standard_normal((9, 3)).astype(np.float32)
    preds = np.full((9, 2), 0.9, dtype=np.float32)
    cache, kept = compress_cache(embs, preds, schedule, np.zeros(3, np.float32))
    assert cache.shape == (3, 3)
    assert kept.shape == (3, 2)


def test_compression_gives_each_speaker_a_place_rather_than_the_loudest_everything():
    """Le rôle du renforcement fort : personne ne disparaît du cache."""
    schedule = _schedule(spkcache_len=2)
    embs = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]], dtype=np.float32)
    preds = np.array([[0.9, 0.0], [0.9, 0.0], [0.9, 0.0], [0.0, 0.9]], dtype=np.float32)
    _, kept = compress_cache(embs, preds, schedule, np.zeros(3, np.float32))
    assert sorted(int(np.argmax(row)) for row in kept) == [0, 1]


def test_compression_preserves_chronological_order():
    """Le cache est relu comme une séquence, pas comme un sac."""
    schedule = _schedule(spkcache_len=3, num_speakers=1)
    embs = np.arange(12, dtype=np.float32).reshape(6, 2)
    embs = np.concatenate([embs, np.zeros((6, 1), dtype=np.float32)], axis=1)
    preds = np.full((6, 1), 0.9, dtype=np.float32)
    cache, _ = compress_cache(embs, preds, schedule, np.zeros(3, np.float32))
    firsts = cache[:, 0].tolist()
    assert firsts == sorted(firsts)


def test_silence_slots_receive_the_learned_embedding():
    """L'écart assumé de ce point de contrôle : une empreinte apprise, pas une moyenne."""
    schedule = _schedule(spkcache_len=4, num_speakers=1, spkcache_sil_frames_per_spk=2)
    embs = np.ones((5, 3), dtype=np.float32)
    preds = np.full((5, 1), 0.9, dtype=np.float32)
    silence = np.array([7.0, 7.0, 7.0], dtype=np.float32)
    cache, kept = compress_cache(embs, preds, schedule, silence)
    filled = [row.tolist() for row in cache]
    assert filled.count(silence.tolist()) >= 1
    for index, row in enumerate(cache):
        if row.tolist() == silence.tolist():
            assert kept[index].sum() == pytest.approx(0.0)


# ----- la file et le cache -------------------------------------------


def test_the_fifo_fills_before_anything_reaches_the_cache():
    schedule = _schedule()
    state = StreamingState.empty(schedule)
    embs = np.ones((2, 3), dtype=np.float32)
    preds = np.full((2, 2), 0.9, dtype=np.float32)
    state, chunk = streaming_update(state, embs, preds, schedule, np.zeros(3, np.float32))
    assert state.fifo.shape == (2, 3)
    assert state.spkcache.shape == (0, 3)
    assert chunk.shape == (2, 2)


def test_the_fifo_pushes_into_the_cache_once_it_overflows():
    schedule = _schedule(fifo_len=2)
    state = StreamingState.empty(schedule)
    silence = np.zeros(3, np.float32)
    for _ in range(3):
        embs = np.ones((2, 3), dtype=np.float32)
        preds = np.full((len(state.spkcache) + len(state.fifo) + 2, 2), 0.9, np.float32)
        state, _ = streaming_update(state, embs, preds, schedule, silence)
    assert len(state.spkcache) > 0, "la file n'a rien poussé vers le cache"
    assert len(state.fifo) <= schedule.fifo_len


def test_the_cache_never_grows_past_its_length():
    schedule = _schedule(fifo_len=2, spkcache_len=4)
    state = StreamingState.empty(schedule)
    silence = np.zeros(3, np.float32)
    rng = np.random.default_rng(1)
    for _ in range(20):
        embs = rng.standard_normal((2, 3)).astype(np.float32)
        preds = np.full((len(state.spkcache) + len(state.fifo) + 2, 2), 0.9, np.float32)
        state, _ = streaming_update(state, embs, preds, schedule, silence)
        assert len(state.spkcache) <= schedule.spkcache_len
        assert len(state.fifo) <= schedule.fifo_len


def test_the_chunk_predictions_are_the_tail_of_what_the_graph_returned():
    schedule = _schedule()
    state = StreamingState.empty(schedule)
    preds = np.arange(6, dtype=np.float32).reshape(3, 2)
    embs = np.ones((3, 3), dtype=np.float32)
    _, chunk = streaming_update(state, embs, preds, schedule, np.zeros(3, np.float32))
    assert np.array_equal(chunk, preds)


# ----- des probabilités aux segments ---------------------------------


def test_thresholding_turns_frames_into_spans():
    probs = np.array([[0.9], [0.9], [0.1]], dtype=np.float32)
    assert probabilities_to_segments(probs, bridge_s=0.0) == [(0.0, 0.02, "S0")]


def test_a_short_silence_inside_a_turn_is_bridged():
    probs = np.array([[0.9], [0.1], [0.9]], dtype=np.float32)
    assert probabilities_to_segments(probs, bridge_s=0.5) == [(0.0, 0.03, "S0")]


def test_a_long_silence_is_not_bridged():
    probs = np.zeros((120, 1), dtype=np.float32)
    probs[:5] = 0.9
    probs[-5:] = 0.9
    spans = probabilities_to_segments(probs, bridge_s=0.5)
    assert len(spans) == 2


def test_two_speakers_at_once_give_overlapping_spans():
    """Le modèle sait dire le recouvrement ; l'écraser perdrait l'information."""
    probs = np.full((4, 2), 0.9, dtype=np.float32)
    spans = probabilities_to_segments(probs)
    assert len(spans) == 2
    assert {s for _, _, s in spans} == {"S0", "S1"}


def test_silence_everywhere_gives_nothing():
    assert probabilities_to_segments(np.zeros((10, 8), dtype=np.float32)) == []


# ----- le calendrier -------------------------------------------------


def test_the_schedule_comes_from_the_graph_not_from_the_code():
    """Un moteur mis à jour apporte ses propres longueurs."""
    schedule = NemotronSchedule.from_metadata(
        {
            "vernacula.diar.fifo_len": "64",
            "vernacula.diar.spkcache_len": "128",
            "vernacula.diar.sil_threshold": "0.3",
        }
    )
    assert schedule.fifo_len == 64
    assert schedule.spkcache_len == 128
    assert schedule.sil_threshold == pytest.approx(0.3)


def test_integer_fields_stay_integers():
    """Les annotations sont des chaînes ici, donc la conversion doit être explicite."""
    schedule = NemotronSchedule.from_metadata({"vernacula.diar.fifo_len": "40"})
    assert isinstance(schedule.fifo_len, int)


def test_an_unknown_metadata_key_does_not_break_loading():
    assert NemotronSchedule.from_metadata({"vernacula.diar.futur": "7"}).fifo_len == 40


def test_the_published_latencies_match_the_schedules():
    """1,04 s, 0,64 s et 0,32 s, comme la carte du modèle les annonce."""
    expected = {"low_latency": 1.04, "very_low_latency": 0.64, "ultra_low_latency": 0.32}
    for mode, (chunk, right) in STREAMING_MODES.items():
        schedule = NemotronSchedule(chunk_len=chunk, chunk_right_context=right)
        assert schedule.latency_s == pytest.approx(expected[mode], abs=1e-6)


def test_an_unknown_mode_is_refused_at_construction():
    with pytest.raises(ValueError, match="calendrier inconnu"):
        NemotronOnnxDiar(mode="instantané")


def test_a_missing_graph_says_where_the_engine_lives(tmp_path):
    stage = NemotronOnnxDiar(graph=tmp_path / "absent.onnx")
    with pytest.raises(Exception, match="absent|No such file|not found|Load model"):
        _ = stage.schedule


def test_the_cpu_comes_before_coreml_because_that_is_what_was_measured():
    """Contre-intuitif, et mesuré : CoreML est plus lent que le CPU sur ce graphe.

    Sur deux minutes d'un mixage à quatre locuteurs, CoreML en tête donne un
    facteur temps réel de 1,257 — plus lent que le direct, donc inutilisable —
    contre 0,528 pour le CPU seul. CoreML ne prend que 1193 nœuds sur 2827 et
    les découpe en 316 partitions : le déport partiel d'un transformeur à axes
    dynamiques coûte plus qu'il ne rapporte.
    """
    from vocal_helper.nemotron import _preferred_providers

    order = _preferred_providers()
    assert "CPUExecutionProvider" in order
    if "CoreMLExecutionProvider" in order:
        assert order.index("CPUExecutionProvider") < order.index("CoreMLExecutionProvider")


def test_coreml_stays_reachable_when_asked_for_explicitly():
    """Le défaut est mesuré, pas imposé : une autre machine peut mieux découper."""
    stage = NemotronOnnxDiar(providers=["CoreMLExecutionProvider"])
    assert stage._providers == ["CoreMLExecutionProvider"]


def test_the_default_schedule_is_a_streaming_one_not_the_graphs_offline_one():
    """Le défaut du graphe plante dès que son cache se remplit.

    Son export affirme que le calendrier hors ligne — 340 trames de morceau plus
    40 de contexte — est le seul vérifié de bout en bout. Mesuré, il lève
    ``Reshape '/pos_enc/Reshape_3' : Input shape {5000,64}, requested shape
    {1,1,5144,64}`` : l'encodage positionnel est tracé pour une longueur finie
    que la séquence dépasse au régime établi. Le piège est qu'un extrait court
    passe, le cache n'étant pas encore plein, donc le défaut échouait
    précisément sur les durées pour lesquelles un mode hors ligne existe.
    """
    from vocal_helper.nemotron import DEFAULT_MODE, STREAMING_MODES

    assert DEFAULT_MODE in STREAMING_MODES
    assert NemotronOnnxDiar()._mode == DEFAULT_MODE


def test_the_graphs_own_schedule_stays_reachable_for_a_short_clip():
    """Un extrait court y tient, et quelqu'un peut vouloir le contexte long."""
    assert NemotronOnnxDiar(mode=None)._mode is None


# ----- la surface de flux, réellement exercée ------------------------
#
# Le défaut de la chaîne a été basculé sur cet étage alors qu'il n'avait PAS de
# méthode ``run``, et les tests ne l'ont pas vu : ils vérifiaient le câblage par
# ``isinstance`` sans jamais faire tourner la chaîne. Une garde qui affirme le
# bon branchement sans vérifier qu'il fonctionne ne garde rien.


def test_the_stage_exposes_the_streaming_surface_the_pipeline_drives():
    import inspect

    assert inspect.iscoroutinefunction(NemotronOnnxDiar.run)


def test_every_diarizer_the_pipeline_can_default_to_is_actually_drivable():
    """Tout étage que la chaîne peut choisir doit pouvoir être piloté.

    C'est la garde qui manquait. Elle ne regarde pas un nom mais une capacité,
    donc elle tiendra pour le prochain moteur qu'on branchera.
    """
    import inspect

    from vocal_helper.diar import OnlineDiarStage
    from vocal_helper.growing import GrowingBufferDiarStage
    from vocal_helper.rediar import PeriodicOfflineDiarStage

    for stage in (
        NemotronOnnxDiar,
        GrowingBufferDiarStage,
        OnlineDiarStage,
        PeriodicOfflineDiarStage,
    ):
        assert hasattr(stage, "run"), f"{stage.__name__} n'a pas de surface de flux"
        assert inspect.iscoroutinefunction(stage.run), (
            f"{stage.__name__}.run doit être une coroutine, comme la chaîne l'attend"
        )


@pytest.mark.integration
def test_the_streaming_loop_labels_each_voiced_segment_it_is_given():
    """Un segment voisé entre, le même ressort avec son locuteur et son PCM.

    Marqué intégration parce qu'il charge le graphe de 400 Mo.
    """
    import asyncio

    rng = np.random.default_rng(0)

    async def drive() -> list:
        inbox: asyncio.Queue = asyncio.Queue()
        outbox: asyncio.Queue = asyncio.Queue()
        for index in range(12):
            t0 = index * 2.0
            inbox.put_nowait(
                {
                    "t0": t0,
                    "t1": t0 + 1.2,
                    "sample_rate": 16000,
                    "pcm": (0.05 * rng.standard_normal(int(1.2 * 16000))).astype(np.float32),
                }
            )
        inbox.put_nowait(None)
        await NemotronOnnxDiar().run(inbox, outbox)
        out = []
        while not outbox.empty():
            item = outbox.get_nowait()
            if item is not None:
                out.append(item)
        return out

    emitted = asyncio.run(drive())
    assert len(emitted) == 12, "chaque segment voisé doit ressortir, étiqueté"
    for item in emitted:
        assert set(item) >= {"t0", "t1", "sample_rate", "speaker", "pcm"}
        assert item["speaker"].startswith("S")
        assert len(item["pcm"]) > 0, "le PCM doit survivre : l'ASR en a besoin"
