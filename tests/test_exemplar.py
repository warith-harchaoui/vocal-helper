"""Le regroupement sans centroïde : exemplaires, meilleur point, reprise globale.

Comme pour le regroupement à centroïdes, tout se vérifie sur des empreintes
fabriquées à la main : pas d'audio, pas de modèle, pas de boucle d'événements.
Ce qui est testé ici est précisément ce qu'une moyenne ne savait pas faire —
tenir une voix à deux régimes, et rendre une fusion globale définissable.
"""

from __future__ import annotations

import numpy as np
import pytest

from vocal_helper.exemplar import Exemplar, ExemplarClusterer, ExemplarPolicy

E0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
E2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
#: Dans la région de E0, mais assez loin pour mériter sa place : distance 0,2.
MODE_B = np.array([0.8, 0.6, 0.0], dtype=np.float32)


def _long(n: int) -> list[float]:
    """``n`` confiances largement au-dessus du plancher d'admission."""
    return [5.0] * n


# ----- la mesure au meilleur point ------------------------------------


def test_distance_to_a_lone_exemplar_is_the_cosine_distance():
    c = ExemplarClusterer()
    c.assign(E0[None, :], _long(1))
    assert c.distances(E0[None, :])[0, 0] == pytest.approx(0.0)
    assert c.distances(E1[None, :])[0, 0] == pytest.approx(1.0)


def test_no_speakers_means_no_columns():
    assert ExemplarClusterer().distances(E0[None, :]).shape == (1, 0)


def test_a_two_mode_voice_is_recognised_by_either_mode():
    """Ce qu'un centroïde ne peut pas faire : être à deux endroits.

    Une moyenne de E0 et MODE_B tomberait entre les deux, donc à ~0,05 de
    chacun ; le meilleur point est à 0 de chacun. L'écart paraît petit ici
    mais c'est lui qui décide d'une jointure au seuil.
    """
    c = ExemplarClusterer()
    c.assign(E0[None, :], _long(1))
    c.assign(MODE_B[None, :], _long(1))
    assert c.exemplar_counts == [2]
    assert c.distances(E0[None, :])[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert c.distances(MODE_B[None, :])[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_neighbours_asks_for_the_kth_nearest_not_the_nearest():
    """Le garde contre le chaînage : un seul exemplaire aberrant ne suffit plus."""
    c = ExemplarClusterer(policy=ExemplarPolicy(neighbours=2))
    c.assign(E0[None, :], _long(1))
    c.assign(MODE_B[None, :], _long(1))
    # Au plus proche, E0 serait à 0 ; au deuxième plus proche, il est à 0,2.
    assert c.distances(E0[None, :])[0, 0] == pytest.approx(0.2, abs=1e-5)


def test_neighbours_beyond_the_memory_falls_back_on_what_exists():
    c = ExemplarClusterer(policy=ExemplarPolicy(neighbours=5))
    c.assign(E0[None, :], _long(1))
    assert c.distances(E0[None, :])[0, 0] == pytest.approx(0.0)


# ----- l'injectivité, conservée du schéma précédent -------------------


def test_two_simultaneous_speakers_never_share_a_label():
    c = ExemplarClusterer()
    assert c.assign(np.stack([E0, E1]), _long(2)) == ["S0", "S1"]
    got = c.assign(np.stack([E0, E0 + 0.01 * E1]), _long(2))
    assert len(set(got)) == 2, f"deux voix simultanées confondues : {got}"


def test_identity_is_tracked_across_a_long_run_of_swaps():
    c = ExemplarClusterer()
    assert c.assign(np.stack([E0, E1]), _long(2)) == ["S0", "S1"]
    for i in range(40):
        pair = [E0, E1] if i % 2 else [E1, E0]
        expected = ["S0", "S1"] if i % 2 else ["S1", "S0"]
        assert c.assign(np.stack(pair), _long(2)) == expected, f"fenêtre {i}"
    assert c.speaker_count == 2


# ----- deviner le compte ----------------------------------------------


def test_a_distant_voice_mints_a_speaker():
    c = ExemplarClusterer()
    c.assign(E0[None, :], _long(1))
    assert c.assign(E1[None, :], _long(1)) == ["S1"]
    assert c.speaker_count == 2


def test_four_orthogonal_voices_are_counted_as_four():
    c = ExemplarClusterer()
    for v in (E0, E1, E2, -E2):
        c.assign(np.asarray(v, dtype=np.float32)[None, :], _long(1))
    assert c.speaker_count == 4


def test_the_cap_forbids_minting_but_keeps_the_window_injective():
    c = ExemplarClusterer(policy=ExemplarPolicy(max_speakers=2))
    c.assign(np.stack([E0, E1]), _long(2))
    got = c.assign(np.stack([E2, E2 + 0.01 * E0]), _long(2))
    assert c.speaker_count == 2
    assert len(set(got)) == 2, f"le plafond a cassé l'injectivité : {got}"


def test_a_saturated_window_still_labels_everyone():
    c = ExemplarClusterer(policy=ExemplarPolicy(max_speakers=2))
    c.assign(np.stack([E0, E1]), _long(2))
    got = c.assign(np.stack([E0, E1, E2]), _long(3))
    assert len(got) == 3
    assert all(g in {"S0", "S1"} for g in got)


# ----- la mémoire : confiance, diversité, éviction --------------------


def test_a_low_confidence_observation_is_labelled_but_not_stored():
    c = ExemplarClusterer()
    c.assign(E0[None, :], _long(1))
    assert c.assign(MODE_B[None, :], [0.05]) == ["S0"]
    assert c.exemplar_counts == [1], "une observation douteuse a pris une place"


def test_a_near_duplicate_reinforces_confidence_instead_of_taking_a_slot():
    c = ExemplarClusterer()
    c.assign(E0[None, :], [1.0])
    c.assign((E0 + 0.01 * E1)[None, :], [9.0])
    assert c.exemplar_counts == [1]
    assert c._sets[0][0].confidence == pytest.approx(9.0)


def test_confidence_is_never_lowered_by_a_weaker_duplicate():
    c = ExemplarClusterer()
    c.assign(E0[None, :], [9.0])
    c.assign((E0 + 0.01 * E1)[None, :], [1.0])
    assert c._sets[0][0].confidence == pytest.approx(9.0)


def test_the_memory_never_exceeds_its_bound():
    c = ExemplarClusterer(policy=ExemplarPolicy(max_exemplars=4, diversity_margin=0.0))
    c.assign(E0[None, :], _long(1))
    rng = np.random.default_rng(0)
    for _ in range(40):
        drift = (E0 + 0.25 * rng.standard_normal(3).astype(np.float32)).astype(np.float32)
        c.assign(drift[None, :], _long(1))
    assert c.exemplar_counts == [4]


def test_eviction_keeps_the_spread_rather_than_the_recent():
    """Évincer le plus redondant, pas le plus ancien : c'est l'étalement qui sert."""
    policy = ExemplarPolicy(max_exemplars=2, diversity_margin=0.0)
    c = ExemplarClusterer(policy=policy)
    c.assign(E0[None, :], _long(1))
    c.assign(MODE_B[None, :], _long(1))  # un second mode, bien séparé
    c.assign((E0 + 0.02 * E1)[None, :], _long(1))  # un quasi-doublon de E0
    assert c.exemplar_counts == [2]
    # Les deux modes doivent avoir survécu, donc les deux restent à distance nulle.
    assert c.distances(MODE_B[None, :])[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert c.distances(E0[None, :])[0, 0] < 0.05


# ----- la reprise globale ---------------------------------------------


def test_two_speakers_split_by_injectivity_are_merged_back():
    """Le mécanisme exact de la sur-segmentation, et sa réparation.

    Deux locuteurs locaux simultanés **doivent** recevoir des étiquettes
    différentes, même quand la segmentation s'est trompée et qu'il n'y en avait
    qu'un. Aucune décision locale ne peut l'éviter ; seule une reprise globale
    le rattrape.
    """
    c = ExemplarClusterer()
    near = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    assert c.assign(np.stack([E0, near]), _long(2)) == ["S0", "S1"]
    assert c.merge() == {"S1": "S0"}
    assert c.speaker_count == 1
    assert c.speaker_labels == ["S0"]


def test_merging_keeps_the_exemplars_of_both():
    c = ExemplarClusterer()
    near = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    c.assign(np.stack([E0, near]), _long(2))
    c.merge()
    assert c.exemplar_counts == [2]


def test_genuinely_distinct_speakers_are_not_merged():
    c = ExemplarClusterer()
    c.assign(np.stack([E0, E1]), _long(2))
    assert c.merge() == {}
    assert c.speaker_count == 2


def test_the_merge_switch_turns_it_off():
    c = ExemplarClusterer(policy=ExemplarPolicy(merge=False))
    near = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    c.assign(np.stack([E0, near]), _long(2))
    assert c.merge() == {}
    assert c.speaker_count == 2


def test_a_chain_of_three_collapses_to_one():
    """La fusion est transitive : A proche de B proche de C donne un locuteur."""
    c = ExemplarClusterer(policy=ExemplarPolicy(merge_threshold=0.3))
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.9, 0.44, 0.0], dtype=np.float32)
    d = np.array([0.65, 0.76, 0.0], dtype=np.float32)
    c.assign(np.stack([a, b, d]), _long(3))
    assert c.speaker_count == 3
    c.merge()
    assert c.speaker_count == 1


def test_the_merge_threshold_governs_how_far_it_reaches():
    near = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    tight = ExemplarClusterer(policy=ExemplarPolicy(merge_threshold=0.0))
    loose = ExemplarClusterer(policy=ExemplarPolicy(merge_threshold=0.5))
    for c in (tight, loose):
        c.assign(np.stack([E0, near]), _long(2))
        c.merge()
    assert tight.speaker_count == 2
    assert loose.speaker_count == 1


def test_a_merged_set_is_pruned_back_to_the_memory_bound():
    policy = ExemplarPolicy(max_exemplars=3, diversity_margin=0.0, merge_threshold=1.5)
    c = ExemplarClusterer(policy=policy)
    c.assign(np.stack([E0, E1, E2]), _long(3))
    assert c.exemplar_counts == [1, 1, 1]
    c.merge()
    assert c.speaker_count == 1
    assert c.exemplar_counts == [3]


def test_merging_is_idempotent():
    c = ExemplarClusterer()
    near = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    c.assign(np.stack([E0, near]), _long(2))
    c.merge()
    assert c.merge() == {}


def test_labels_are_never_reused_after_a_merge():
    """Une étiquette absorbée ne doit pas resservir : elle désignerait deux voix."""
    c = ExemplarClusterer()
    near = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    c.assign(np.stack([E0, near]), _long(2))
    c.merge()
    assert c.assign(E2[None, :], _long(1)) == ["S2"]
    assert c.speaker_labels == ["S0", "S2"]


# ----- validation des entrées -----------------------------------------


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="durées"):
        ExemplarClusterer().assign(np.stack([E0, E1]), [1.0])


def test_a_one_dimensional_input_is_refused():
    with pytest.raises(ValueError, match=r"\(k, d\)"):
        ExemplarClusterer().assign(E0, [1.0])


def test_an_empty_window_is_not_an_error():
    assert ExemplarClusterer().assign(np.zeros((0, 3), dtype=np.float32), []) == []


def test_unnormalised_input_is_normalised():
    c = ExemplarClusterer()
    c.assign((7.0 * E0)[None, :], _long(1))
    assert float(np.linalg.norm(c._sets[0][0].vector)) == pytest.approx(1.0, abs=1e-5)


def test_a_zero_vector_does_not_divide_by_zero():
    c = ExemplarClusterer()
    assert c.assign(np.zeros((1, 3), dtype=np.float32), _long(1)) == ["S0"]


def test_an_exemplar_carries_its_confidence():
    e = Exemplar(vector=E0, confidence=3.5)
    assert e.confidence == pytest.approx(3.5)


# ----- la référence : un point observé, jamais une moyenne ------------


def test_the_confidence_reference_is_the_best_observed_point():
    """La règle du magasin biométrique : la ressemblance au meilleur indice."""
    c = ExemplarClusterer(policy=ExemplarPolicy(reference="confiance"))
    c.assign(E0[None, :], [1.0])
    c.assign(MODE_B[None, :], [9.0])  # mieux attestée, elle devient la référence
    assert c.exemplar_counts == [2]
    assert c.distances(MODE_B[None, :])[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert c.distances(E0[None, :])[0, 0] == pytest.approx(0.2, abs=1e-5)


def test_the_confidence_reference_has_no_optimistic_bias():
    """Le défaut mesuré du meilleur point : son minimum baisse avec la taille.

    Un point observé unique ne bouge pas quand l'ensemble grandit, donc un seul
    exemplaire bruité ne peut plus capturer un locuteur.
    """
    stable = ExemplarClusterer(policy=ExemplarPolicy(reference="confiance"))
    optimistic = ExemplarClusterer(policy=ExemplarPolicy(reference="meilleur"))
    probe = np.array([0.0, 0.6, 0.8], dtype=np.float32)
    rng = np.random.default_rng(1)
    for c in (stable, optimistic):
        c.assign(E0[None, :], [9.0])
        before = float(c.distances(probe[None, :])[0, 0])
        for _ in range(20):
            noisy = (E0 + 0.5 * rng.standard_normal(3).astype(np.float32)).astype(np.float32)
            c.assign(noisy[None, :], [1.0])
        c._before, c._after = before, float(c.distances(probe[None, :])[0, 0])
    assert stable._after == pytest.approx(stable._before, abs=1e-6)
    assert optimistic._after < optimistic._before - 0.05


def test_the_medoid_reference_resists_a_lone_intruder():
    """Le médoïde ne peut pas être l'intrus : il faut être central pour l'être.

    L'intrus est ici à portée de jointure — sinon il formerait son propre
    locuteur et il n'y aurait rien à tester — et il est le mieux attesté des
    trois, donc la règle de confiance le choisirait. Le médoïde, non.
    """
    intruder = np.array([0.5, 0.866, 0.0], dtype=np.float32)  # à 0,5 de E0
    policy = ExemplarPolicy(reference="médoïde", diversity_margin=0.0)
    c = ExemplarClusterer(policy=policy)
    c.assign(E0[None, :], [1.0])
    c.assign((E0 + 0.2 * E1)[None, :], [1.0])
    assert c.assign(intruder[None, :], [50.0]) == ["S0"], "l'intrus devait rejoindre"
    assert c.exemplar_counts == [3]
    assert c.distances(E0[None, :])[0, 0] < 0.1, "le médoïde a suivi l'intrus"


def test_confidence_and_medoid_disagree_where_it_matters():
    """Pourquoi le médoïde est le repli : la confiance seule peut être trompée."""
    intruder = np.array([0.5, 0.866, 0.0], dtype=np.float32)
    built = {}
    for mode in ("confiance", "médoïde"):
        c = ExemplarClusterer(policy=ExemplarPolicy(reference=mode, diversity_margin=0.0))
        c.assign(E0[None, :], [1.0])
        c.assign((E0 + 0.2 * E1)[None, :], [1.0])
        c.assign(intruder[None, :], [50.0])
        built[mode] = float(c.distances(E0[None, :])[0, 0])
    assert built["confiance"] == pytest.approx(0.5, abs=1e-3)
    assert built["médoïde"] < 0.1


def test_an_unknown_reference_is_refused():
    c = ExemplarClusterer(policy=ExemplarPolicy(reference="moyenne"))
    c.assign(E0[None, :], _long(1))
    with pytest.raises(ValueError, match="référence inconnue"):
        c.distances(E1[None, :])
