"""Le regroupement incrémental, testé sans audio ni modèle.

Tout l'intérêt d'avoir écrit l'algorithme sur place plutôt que d'importer
diart est là : les trois garde-fous (injectivité, ``rho_update``, ``delta_new``) sont
de l'arithmétique vectorielle, donc chacun se vérifie sur des empreintes
fabriquées à la main, en millisecondes et sans télécharger un modèle.
"""

from __future__ import annotations

import numpy as np
import pytest

from vocal_helper.incremental import (
    IncrementalClusterer,
    IncrementalDiarStage,
    IncrementalPolicy,
    _votes_to_segments,
    cosine_distances,
)

E0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
E2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _long(n: int) -> list[float]:
    """``n`` durées confortablement au-dessus de ``rho_update_s``."""
    return [5.0] * n


# ----- distances ------------------------------------------------------


def test_cosine_distances_shape_and_values():
    d = cosine_distances(np.stack([E0, E1]), np.stack([E0, E1, E2]))
    assert d.shape == (2, 3)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, 1] == pytest.approx(1.0)
    assert d[1, 1] == pytest.approx(0.0)


def test_cosine_distance_is_two_for_opposites():
    assert cosine_distances(E0[None, :], (-E0)[None, :])[0, 0] == pytest.approx(2.0)


# ----- la contrainte d'injectivité ------------------------------------


def test_two_simultaneous_speakers_never_share_a_label():
    """Le défaut qui faisait s'effondrer l'étage glouton sur une étiquette."""
    c = IncrementalClusterer()
    assert c.assign(np.stack([E0, E1]), _long(2)) == ["S0", "S1"]
    # Deux empreintes presque identiques, actives en même temps : le glouton
    # les aurait mises sur le même centroïde. L'algorithme hongrois ne peut pas.
    near = np.stack([E0, E0 + 0.01 * E1])
    got = c.assign(near, _long(2))
    assert len(set(got)) == 2, f"deux voix simultanées confondues : {got}"


def test_label_order_follows_identity_not_position():
    c = IncrementalClusterer()
    c.assign(np.stack([E0, E1]), _long(2))
    assert c.assign(np.stack([E1, E0]), _long(2)) == ["S1", "S0"]


def test_assignment_survives_a_long_sequence_of_swaps():
    """La dérive, mesurée là où elle se produit : au fil des fenêtres."""
    c = IncrementalClusterer()
    # La première fenêtre *définit* la correspondance ; on la pose à part pour
    # que la suite teste ce qui nous intéresse, la tenue de cette correspondance.
    assert c.assign(np.stack([E0, E1]), _long(2)) == ["S0", "S1"]
    for i in range(50):
        pair = [E0, E1] if i % 2 else [E1, E0]
        got = c.assign(np.stack(pair), _long(2))
        expected = ["S0", "S1"] if i % 2 else ["S1", "S0"]
        assert got == expected, f"fenêtre {i} : {got}"
    assert c.speaker_count == 2


# ----- delta_new : deviner le compte ----------------------------------


def test_a_distant_voice_mints_a_new_speaker():
    c = IncrementalClusterer()
    c.assign(E0[None, :], _long(1))
    assert c.assign(E1[None, :], _long(1)) == ["S1"]
    assert c.speaker_count == 2


def test_a_close_voice_joins_rather_than_minting():
    c = IncrementalClusterer()
    c.assign(E0[None, :], _long(1))
    assert c.assign((E0 + 0.05 * E1)[None, :], _long(1)) == ["S0"]
    assert c.speaker_count == 1


def test_delta_new_is_the_knob_that_sets_the_count():
    tight = IncrementalClusterer(policy=IncrementalPolicy(delta_new=0.01))
    loose = IncrementalClusterer(policy=IncrementalPolicy(delta_new=1.5))
    probe = (E0 + 0.5 * E1)[None, :]
    for c in (tight, loose):
        c.assign(E0[None, :], _long(1))
        c.assign(probe, _long(1))
    assert tight.speaker_count == 2
    assert loose.speaker_count == 1


def test_speaker_count_is_guessed_not_configured():
    """Quatre voix orthogonales, aucun compte fourni : le flux en trouve quatre."""
    c = IncrementalClusterer()
    voices = [E0, E1, E2, np.array([0.0, 0.0, -1.0], dtype=np.float32)]
    for v in voices:
        c.assign(v[None, :], _long(1))
    assert c.speaker_count == 4
    assert c.speaker_labels == ["S0", "S1", "S2", "S3"]


# ----- le plafond -----------------------------------------------------


def test_the_cap_forbids_minting_and_falls_back_to_the_nearest():
    c = IncrementalClusterer(policy=IncrementalPolicy(max_speakers=2))
    c.assign(np.stack([E0, E1]), _long(2))
    assert c.assign(E2[None, :], _long(1))[0] in {"S0", "S1"}
    assert c.speaker_count == 2


def test_the_cap_still_keeps_a_window_injective():
    c = IncrementalClusterer(policy=IncrementalPolicy(max_speakers=2))
    c.assign(np.stack([E0, E1]), _long(2))
    got = c.assign(np.stack([E2, E2 + 0.01 * E0]), _long(2))
    assert len(set(got)) == 2, f"le plafond a cassé l'injectivité : {got}"


def test_a_saturated_window_assigns_everyone_anyway():
    """Plus de locuteurs simultanés que le plafond : personne n'est perdu."""
    c = IncrementalClusterer(policy=IncrementalPolicy(max_speakers=2))
    c.assign(np.stack([E0, E1]), _long(2))
    got = c.assign(np.stack([E0, E1, E2]), _long(3))
    assert len(got) == 3
    assert all(g in {"S0", "S1"} for g in got)


# ----- rho_update -----------------------------------------------------


def test_a_short_clean_share_assigns_but_does_not_write():
    c = IncrementalClusterer()
    c.assign(E0[None, :], _long(1))
    before = c.centroids().copy()
    assert c.assign((E0 + 0.3 * E1)[None, :], [0.05]) == ["S0"]
    np.testing.assert_allclose(c.centroids(), before)


def test_a_long_clean_share_does_write():
    c = IncrementalClusterer()
    c.assign(E0[None, :], _long(1))
    before = c.centroids().copy()
    c.assign((E0 + 0.3 * E1)[None, :], _long(1))
    assert not np.allclose(c.centroids(), before)


def test_rho_update_protects_the_centroid_from_a_run_of_blends():
    """Le mécanisme exact qui manquait : la contamination par recouvrement."""
    guarded = IncrementalClusterer()
    exposed = IncrementalClusterer(policy=IncrementalPolicy(rho_update_s=0.0))
    blend = (E0 + 0.9 * E1)[None, :]
    for c in (guarded, exposed):
        c.assign(E0[None, :], _long(1))
        for _ in range(20):
            c.assign(blend, [0.05])
    assert cosine_distances(E0[None, :], guarded.centroids())[0, 0] == pytest.approx(0.0)
    assert cosine_distances(E0[None, :], exposed.centroids())[0, 0] > 0.1


# ----- validation des entrées -----------------------------------------


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="durées"):
        IncrementalClusterer().assign(np.stack([E0, E1]), [1.0])


def test_a_one_dimensional_input_is_refused():
    with pytest.raises(ValueError, match=r"\(k, d\)"):
        IncrementalClusterer().assign(E0, [1.0])


def test_an_empty_window_is_not_an_error():
    assert IncrementalClusterer().assign(np.zeros((0, 3), dtype=np.float32), []) == []


def test_unnormalised_input_is_normalised():
    c = IncrementalClusterer()
    c.assign((7.0 * E0)[None, :], _long(1))
    assert float(np.linalg.norm(c.centroids()[0])) == pytest.approx(1.0, abs=1e-5)


def test_a_zero_vector_does_not_divide_by_zero():
    c = IncrementalClusterer()
    assert c.assign(np.zeros((1, 3), dtype=np.float32), _long(1)) == ["S0"]


# ----- l'étage : fenêtrage, part propre, vote -------------------------


def test_windows_cover_the_whole_stream_including_its_end():
    stage = IncrementalDiarStage(window_s=5.0, step_s=2.5)
    wins = stage.windows(12.0)
    assert wins[0][0] == 0.0
    assert wins[-1][1] == pytest.approx(12.0)
    assert all(b - a == pytest.approx(5.0) for a, b in wins)


def test_a_stream_shorter_than_a_window_is_one_window():
    assert IncrementalDiarStage(window_s=5.0).windows(2.0) == [(0.0, 2.0)]


def test_windows_overlap_by_the_step():
    wins = IncrementalDiarStage(window_s=4.0, step_s=1.0).windows(10.0)
    assert wins[1][0] - wins[0][0] == pytest.approx(1.0)


def test_a_step_longer_than_the_window_is_refused():
    with pytest.raises(ValueError, match="dépasse"):
        IncrementalDiarStage(window_s=2.0, step_s=3.0)


def test_non_positive_geometry_is_refused():
    with pytest.raises(ValueError, match="positifs"):
        IncrementalDiarStage(window_s=0.0)


def test_clean_spans_remove_every_overlap():
    stage = IncrementalDiarStage()
    local = [(0.0, 3.0, "A"), (1.0, 2.0, "B")]
    assert stage._clean_spans(local, "A") == [(0.0, 1.0), (2.0, 3.0)]


def test_a_fully_overlapped_speaker_has_no_clean_share():
    stage = IncrementalDiarStage()
    local = [(1.0, 2.0, "A"), (0.0, 3.0, "B")]
    assert stage._clean_spans(local, "A") == []


def test_clean_spans_of_a_lone_speaker_are_its_own_segments():
    stage = IncrementalDiarStage()
    assert stage._clean_spans([(0.0, 1.0, "A")], "A") == [(0.0, 1.0)]


def test_votes_pick_the_majority_label_per_frame():
    votes = {
        "S0": np.array([3.0, 1.0], dtype=np.float32),
        "S1": np.array([1.0, 3.0], dtype=np.float32),
    }
    assert _votes_to_segments(votes, 10.0) == [(0.0, 0.1, "S0"), (0.1, 0.2, "S1")]


def test_frames_nobody_voted_for_stay_silent():
    votes = {"S0": np.array([1.0, 0.0, 1.0], dtype=np.float32)}
    assert _votes_to_segments(votes, 10.0) == [(0.0, 0.1, "S0"), (0.2, 0.3, "S0")]


def test_no_votes_is_an_empty_diarization():
    assert _votes_to_segments({}, 10.0) == []


def test_contiguous_frames_of_one_speaker_merge_into_one_segment():
    votes = {"S0": np.ones(5, dtype=np.float32)}
    assert _votes_to_segments(votes, 10.0) == [(0.0, 0.5, "S0")]


def test_max_speakers_passed_to_diarize_overrides_the_policy():
    class _Mute:
        """Un segmenteur qui n'entend rien : évite de charger un modèle."""

        def diarize(self, pcm, sr, **kwargs):
            return []

    stage = IncrementalDiarStage(policy=IncrementalPolicy(max_speakers=None))
    stage._segmenter, stage._embedder = _Mute(), object()
    stage.diarize(np.zeros(10, dtype=np.float32), 16000, max_speakers=3)
    assert stage.policy.max_speakers == 3


def test_the_stage_reports_what_it_discovered():
    stage = IncrementalDiarStage()
    stage.clusterer.assign(np.stack([E0, E1]), _long(2))
    assert stage.discovered_speakers == 2
    assert stage.speaker_labels == ["S0", "S1"]
