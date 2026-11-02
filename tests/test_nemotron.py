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


def test_the_default_is_the_published_low_latency_schedule():
    """Le défaut est celui que la provenance publie, et voici pourquoi.

    J'ai d'abord pris le plus long contexte viable, 6,4 s, parce qu'il est moins
    cher à qualité égale : 0,166 de facteur temps réel contre 0,258, pour 0,092
    de taux d'erreur contre 0,090. Mais cette égalité n'existe qu'en salle
    **anéchoïque**, et une salle de réunion ne l'est pas. Réverbérée à 0,6 s de
    temps de décroissance, les deux se séparent nettement : 0,350 contre 0,252 de
    taux d'erreur, 0,190 contre 0,151 de confusion, et un accord longue portée de
    0,372 contre 0,668 — presque doublé.

    Le contexte long dilue donc la décision quand la salle smeare le signal : il
    voit plus de matière, mais une matière plus confuse. Le défaut est donc le
    calendrier validé par la provenance, ce qui aurait dû être l'hypothèse de
    départ.
    """
    from vocal_helper.nemotron import DEFAULT_LATENCY_S

    assert NemotronOnnxDiar().schedule.latency_s == pytest.approx(DEFAULT_LATENCY_S)
    assert (DEFAULT_LATENCY_S, 0) in [
        (NemotronSchedule(chunk_len=c, chunk_right_context=r).latency_s, 0)
        for c, r in STREAMING_MODES.values()
    ], "le défaut doit être un calendrier publié, pas une valeur inventée"


def test_asking_for_no_budget_takes_the_longest_context_the_fifo_allows():
    """Le défaut prend le plus long contexte VIABLE, et la file est ce qui borne.

    Deux limites se superposent, et c'est la seconde qui mord. La séquence que le
    graphe accepte en est une : 625 trames, cache et file comprises. Mais la mise
    à jour ajoute ``chunk_len`` trames à la file et n'en retire que ``fifo_len``,
    donc un morceau plus long que la file la fait grandir sans borne — avec 281
    de morceau elle monte à 241 au lieu de rester à 40, et la séquence atteint
    826 trames. C'est le portage fidèle de la référence, dont la règle n'est
    cohérente que si le morceau n'excède pas la file : ses trois calendriers de
    flux publiés le respectent et son calendrier hors ligne, à 340 trames, le
    viole — ce qui est exactement pourquoi ce dernier ne fonctionne pas.

    Le défaut est donc un morceau de ``fifo_len`` trames, soit 6,4 s de latence :
    six fois le calendrier à basse latence, et non les vingt-cinq que la seule
    capacité de séquence laissait espérer.
    """
    from vocal_helper.nemotron import fits_capacity

    schedule = NemotronOnnxDiar(latency_s=None).schedule
    assert fits_capacity(schedule)
    assert schedule.chunk_len <= schedule.fifo_len, (
        "un morceau plus long que la file la fait grandir sans borne"
    )
    assert schedule.latency_s > 5.0, "sans contrainte, le contexte doit être long"


def test_a_chunk_longer_than_the_fifo_is_never_chosen():
    """La contrainte qui a réellement fait échouer le défaut précédent."""
    from vocal_helper.nemotron import NemotronSchedule, longest_schedule_within

    schedule = NemotronSchedule()
    for budget in (None, 60.0, 25.0, 6.4, 1.04, 0.32):
        chunk, _ = longest_schedule_within(schedule, latency_s=budget)
        assert chunk <= schedule.fifo_len, f"budget {budget} choisit un morceau de {chunk}"


def test_a_latency_budget_is_the_only_knob_needed():
    """Un budget en secondes, pas un nom de calendrier à connaître."""
    assert NemotronOnnxDiar(latency_s=1.04).schedule.latency_s == pytest.approx(1.04)
    assert NemotronOnnxDiar(latency_s=0.32).schedule.latency_s == pytest.approx(0.32)


def test_a_named_schedule_still_wins_for_reproducing_a_published_setup():
    assert NemotronOnnxDiar(mode="ultra_low_latency").schedule.latency_s == pytest.approx(0.32)


def test_the_offline_schedule_is_refused_with_its_arithmetic_spelled_out():
    """Mieux vaut refuser en expliquant que planter sur un nœud Reshape."""
    from vocal_helper.nemotron import NemotronSchedule, fits_capacity

    assert not fits_capacity(NemotronSchedule(chunk_len=340, chunk_right_context=40))


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


# ----- le budget en direct -------------------------------------------


def test_a_conversational_agent_can_afford_the_low_latency_schedule():
    """La question qui décide pour ``discussion-helper``.

    Un agent qui répond dans l'instant ne peut rien démentir, donc il veut le
    calendrier à basse latence et non le contexte long. Ce qu'il doit savoir
    n'est pas la latence d'un étage mais le **facteur temps réel total** du
    système, puisque les étages travaillent à des cadences différentes : ce qui
    s'additionne est le travail sur une même unité de calcul, et ce qui se
    recouvre est le travail sur des unités distinctes.

    Mesuré : 0,268 avec la transcription sur Metal, 0,432 sans aucune
    accélération. Il reste donc de la marge dans les deux cas, et c'est ce qui
    rend le choix possible.
    """
    from vocal_helper.accel import StageLoad, live_budget

    for asr_unit, asr_rtf in (("metal", 0.033), ("cpu", 0.164)):
        budget = live_budget(
            [
                StageLoad("vad_silero", "cpu", 0.01),
                StageLoad("diar_nemotron", "cpu", 0.258),  # 1,04 s, mesuré
                StageLoad("asr_whisper", asr_unit, asr_rtf),
            ]
        )
        assert budget.effective_rtf < 1.0, (
            f"à basse latence avec l'ASR sur {asr_unit}, le système ne tient pas "
            f"le direct : {budget.effective_rtf:.3f}"
        )


def test_the_long_context_default_is_cheaper_than_low_latency():
    """Le contexte long n'est pas un luxe : il coûte moins.

    0,166 de facteur temps réel contre 0,258, pour 0,092 de taux d'erreur contre
    0,090 — identique au bruit près. C'est pourquoi c'est le défaut, et pourquoi
    la basse latence est un choix à faire et non un défaut à subir.
    """
    from vocal_helper.accel import StageLoad, live_budget

    def total(diar_rtf: float) -> float:
        return live_budget(
            [
                StageLoad("vad_silero", "cpu", 0.01),
                StageLoad("diar_nemotron", "cpu", diar_rtf),
                StageLoad("asr_whisper", "metal", 0.033),
            ]
        ).effective_rtf

    assert total(0.166) < total(0.258)


def test_the_pipeline_exposes_one_latency_knob_and_it_reaches_the_engine():
    """Un budget en secondes au niveau de la chaîne, pas un nom de calendrier."""
    import vocal_helper as voh

    async def source():
        yield {"t0": 0.0, "sample_rate": 16000, "pcm": np.zeros(320, dtype=np.float32)}

    default = voh.Pipeline(source=source, config=voh.PipelineConfig(llm=None))
    quick = voh.Pipeline(source=source, config=voh.PipelineConfig(llm=None, diar_latency_s=1.04))
    if isinstance(default._diar, NemotronOnnxDiar):
        assert default._diar.schedule.latency_s > quick._diar.schedule.latency_s
        assert quick._diar.schedule.latency_s == pytest.approx(1.04)


def test_a_named_provider_is_read_tolerantly() -> None:
    """Ce nom s'écrit dans un shell, pas dans du code : la graphie exacte du
    runtime ne doit pas être exigée."""
    from vocal_helper.nemotron import provider_from_name

    for forme in ("coreml", "CoreML", "core-ml", "CoreMLExecutionProvider"):
        assert provider_from_name(forme) == "CoreMLExecutionProvider", forme
    assert provider_from_name("dml") == "DmlExecutionProvider"
    assert provider_from_name("metal") is None


def test_an_unknown_provider_name_raises_instead_of_falling_back(monkeypatch) -> None:
    """Un repli rendrait une faute de frappe indétectable : on croirait tourner
    sur CUDA en tournant sur le CPU, et la seule trace serait un RTF décevant."""
    from vocal_helper import nemotron

    monkeypatch.setenv(nemotron.PROVIDER_ENV, "cudaa")
    with pytest.raises(ValueError, match="ne désigne aucun moteur"):
        nemotron._requested_providers()


def test_a_provider_absent_from_this_install_raises_too(monkeypatch) -> None:
    """ONNX Runtime, lui, se rabat sur le CPU en écrivant un avertissement que
    personne ne lit — exactement le repli silencieux qu'on s'interdit."""
    from vocal_helper import nemotron

    monkeypatch.setenv(nemotron.PROVIDER_ENV, "qnn")
    with pytest.raises(ValueError, match="absent de cette installation"):
        nemotron._requested_providers()


def test_no_environment_means_no_opinion(monkeypatch) -> None:
    from vocal_helper import nemotron

    monkeypatch.delenv(nemotron.PROVIDER_ENV, raising=False)
    assert nemotron._requested_providers() is None
    monkeypatch.setenv(nemotron.PROVIDER_ENV, "   ")
    assert nemotron._requested_providers() is None
