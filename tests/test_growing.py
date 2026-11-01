"""La fenêtre croissante : stabilité des étiquettes, calendrier, révisions.

Tout ce qui décide du comportement se vérifie sans modèle : le diariseur
différé est remplacé par des bouchons qui rendent ce qu'on veut éprouver —
des étiquettes qui changent de nom, un locuteur qui arrive en retard, deux
locuteurs qui échangent leurs noms d'une passe à l'autre.
"""

from __future__ import annotations

import numpy as np
import pytest

from vocal_helper.growing import (
    GrowingBufferDiarStage,
    GrowingBufferPolicy,
    map_by_overlap,
    refresh_times,
)

SR = 16000


def _audio(seconds: float) -> np.ndarray:
    """Du silence : les bouchons ne l'écoutent pas."""
    return np.zeros(int(seconds * SR), dtype=np.float32)


class _Scripted:
    """Un diariseur qui rend, passe après passe, ce qu'on lui a écrit."""

    def __init__(self, script: list[list[tuple[float, float, str]]]) -> None:
        self.script = script
        self.calls = 0

    def diarize(self, pcm, sr, **kwargs):  # noqa: ANN001, ANN003
        out = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return out


# ----- l'appariement -------------------------------------------------


def test_the_mapping_follows_time_not_names():
    assert map_by_overlap(
        [(0.0, 2.0, "a"), (2.0, 4.0, "b")], [(0.0, 2.0, "y"), (2.0, 4.0, "x")]
    ) == {"a": "y", "b": "x"}


def test_nothing_shared_maps_nothing():
    assert map_by_overlap([(0.0, 1.0, "a")], [(9.0, 10.0, "z")]) == {}


def test_two_labels_never_collapse_onto_one():
    """Un argmax par étiquette le ferait, et fusionnerait deux locuteurs."""
    left = [(0.0, 10.0, "a"), (0.0, 9.0, "b")]
    right = [(0.0, 10.0, "x"), (0.0, 1.0, "y")]
    got = map_by_overlap(left, right)
    assert len(set(got.values())) == len(got)


def test_an_empty_side_maps_nothing():
    assert map_by_overlap([], [(0.0, 1.0, "x")]) == {}


# ----- le calendrier -------------------------------------------------


def test_the_schedule_grows_geometrically_and_ends_at_the_duration():
    times = refresh_times(100.0, first_s=10.0, ratio=2.0)
    assert times == [10.0, 20.0, 40.0, 80.0, 100.0]


def test_a_stream_shorter_than_the_first_pass_gets_one_pass():
    assert refresh_times(8.0, first_s=10.0, ratio=2.0) == [8.0]


def test_a_cap_stops_the_wait_from_growing():
    times = refresh_times(100.0, first_s=10.0, ratio=2.0, max_period_s=25.0)
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert max(gaps) <= 25.0 + 1e-6


def test_no_duration_means_no_pass():
    assert refresh_times(0.0, first_s=10.0, ratio=2.0) == []


def test_the_cost_stays_proportional_not_quadratic():
    """C'est tout l'intérêt du pas géométrique : chaque passe rediarise tout."""
    duration = 3600.0
    times = refresh_times(duration, first_s=60.0, ratio=1.6)
    assert sum(times) / duration < 5.0, "le coût doit rester un petit multiple"


def test_the_worst_staleness_is_reported_honestly():
    policy = GrowingBufferPolicy(first_s=60.0, ratio=1.6)
    assert policy.worst_staleness_s(1200.0) == pytest.approx(377.487, abs=0.01)


def test_capping_the_period_shortens_the_staleness():
    duration = 3600.0
    loose = GrowingBufferPolicy(first_s=60.0, ratio=1.6)
    tight = GrowingBufferPolicy(first_s=60.0, ratio=1.6, max_period_s=120.0)
    assert tight.worst_staleness_s(duration) < loose.worst_staleness_s(duration)


# ----- la stabilité des étiquettes -----------------------------------


def test_a_speaker_keeps_its_name_when_the_engine_renames_its_cluster():
    """Le cœur du mécanisme : les noms de grappes sont libres, l'identité non."""
    stage = GrowingBufferDiarStage(
        offline=_Scripted(
            [
                [(0.0, 10.0, "cluster_7")],
                [(0.0, 20.0, "cluster_3")],
                [(0.0, 40.0, "autre_chose")],
            ]
        ),
        policy=GrowingBufferPolicy(first_s=10.0, ratio=2.0),
    )
    stage.diarize(_audio(40.0), SR)
    assert stage.speaker_labels == ["P0"], "le locuteur a changé de nom en route"
    assert stage.discovered_speakers == 1


def test_two_speakers_swapping_cluster_names_are_not_swapped_for_the_reader():
    stage = GrowingBufferDiarStage(
        offline=_Scripted(
            [
                [(0.0, 5.0, "a"), (5.0, 10.0, "b")],
                [(0.0, 5.0, "b"), (5.0, 10.0, "a")],
            ]
        ),
        policy=GrowingBufferPolicy(first_s=10.0, ratio=2.0),
    )
    stage.diarize(_audio(10.0), SR)
    first, second = stage.segments()
    assert first[2] != second[2]
    assert stage.discovered_speakers == 2


def test_a_latecomer_is_discovered_rather_than_forced_onto_someone():
    stage = GrowingBufferDiarStage(
        offline=_Scripted(
            [
                [(0.0, 10.0, "a")],
                [(0.0, 10.0, "a"), (10.0, 20.0, "b")],
            ]
        ),
        policy=GrowingBufferPolicy(first_s=10.0, ratio=2.0),
    )
    stage.diarize(_audio(20.0), SR)
    assert stage.discovered_speakers == 2


def test_the_first_pass_revises_nothing():
    stage = GrowingBufferDiarStage(
        offline=_Scripted([[(0.0, 10.0, "a")]]),
        policy=GrowingBufferPolicy(first_s=10.0, ratio=2.0),
    )
    assert stage.refresh(_audio(10.0), SR) == []


def test_a_contradicting_pass_says_so_instead_of_rewriting_in_silence():
    stage = GrowingBufferDiarStage(
        offline=_Scripted(
            [
                [(0.0, 10.0, "a"), (10.0, 20.0, "a")],
                [(0.0, 10.0, "a"), (10.0, 20.0, "b")],
            ]
        ),
        policy=GrowingBufferPolicy(first_s=10.0, ratio=2.0),
    )
    stage.refresh(_audio(20.0), SR)
    revisions = stage.refresh(_audio(20.0), SR)
    assert len(revisions) == 1
    assert revisions[0].old_speaker != revisions[0].new_speaker


def test_an_engine_that_says_nothing_leaves_the_stream_untouched():
    stage = GrowingBufferDiarStage(
        offline=_Scripted([[]]), policy=GrowingBufferPolicy(first_s=10.0, ratio=2.0)
    )
    assert stage.diarize(_audio(20.0), SR) == []
    assert stage.discovered_speakers == 0


def test_the_passes_are_counted_even_when_fruitless():
    stage = GrowingBufferDiarStage(
        offline=_Scripted([[]]), policy=GrowingBufferPolicy(first_s=10.0, ratio=2.0)
    )
    stage.diarize(_audio(40.0), SR)
    assert stage.passes == len(GrowingBufferPolicy(first_s=10.0, ratio=2.0).times(40.0))


def test_resetting_forgets_the_stream_but_keeps_the_configuration():
    policy = GrowingBufferPolicy(first_s=10.0, ratio=2.0)
    stage = GrowingBufferDiarStage(offline=_Scripted([[(0.0, 10.0, "a")]]), policy=policy)
    stage.diarize(_audio(10.0), SR)
    stage.reset()
    assert stage.discovered_speakers == 0
    assert stage.segments() == []
    assert stage.policy is policy


# ----- le filtre des locuteurs fugaces -------------------------------


def test_a_fleeting_label_is_not_published():
    """Un locuteur d'une seconde est en général un bruit pris pour une voix."""
    stage = GrowingBufferDiarStage(
        offline=_Scripted([[(0.0, 30.0, "a"), (30.0, 30.4, "b")]]),
        policy=GrowingBufferPolicy(first_s=10.0, ratio=2.0, min_speaker_s=2.0),
    )
    stage.refresh(_audio(31.0), SR)
    assert stage.speaker_labels == ["P0"]


def test_without_the_filter_every_label_is_published():
    stage = GrowingBufferDiarStage(
        offline=_Scripted([[(0.0, 30.0, "a"), (30.0, 30.4, "b")]]),
        policy=GrowingBufferPolicy(first_s=10.0, ratio=2.0, min_speaker_s=0.0),
    )
    stage.refresh(_audio(31.0), SR)
    assert stage.discovered_speakers == 2


# ----- les indices de compte -----------------------------------------


def test_the_count_hints_reach_the_engine():
    seen: dict[str, object] = {}

    class _Nosy:
        def diarize(self, pcm, sr, **kwargs):  # noqa: ANN001, ANN003
            seen.update(kwargs)
            return [(0.0, 10.0, "a")]

    stage = GrowingBufferDiarStage(
        offline=_Nosy(), policy=GrowingBufferPolicy(first_s=10.0, ratio=2.0)
    )
    stage.diarize(_audio(10.0), SR, max_speakers=4)
    assert seen == {"max_speakers": 4}


# ----- la surface de flux --------------------------------------------


async def _drain(stage, segments: list[tuple[float, float]], sr: int = SR) -> list:
    """Faire passer des segments voisés dans l'étage et récolter la sortie."""
    import asyncio

    inbox: asyncio.Queue = asyncio.Queue()
    outbox: asyncio.Queue = asyncio.Queue()
    for t0, t1 in segments:
        inbox.put_nowait(
            {
                "t0": t0,
                "t1": t1,
                "sample_rate": sr,
                "pcm": np.zeros(int((t1 - t0) * sr), dtype=np.float32),
            }
        )
    inbox.put_nowait(None)
    await stage.run(inbox, outbox)
    out = []
    while not outbox.empty():
        item = outbox.get_nowait()
        if item is not None:
            out.append(item)
    return out


def test_the_stream_closes_with_a_final_pass_over_everything():
    """La fin de séance n'a jamais été rediarisée, et c'est ce qu'on regarde."""
    import asyncio

    engine = _Scripted([[(0.0, 30.0, "a")]])
    stage = GrowingBufferDiarStage(
        offline=engine, policy=GrowingBufferPolicy(first_s=1000.0, ratio=2.0)
    )
    asyncio.run(_drain(stage, [(0.0, 10.0), (12.0, 20.0)]))
    assert engine.calls >= 1, "la fermeture doit déclencher une passe"
    assert stage.discovered_speakers == 1


def test_the_online_stage_publishes_before_the_first_pass():
    """Sans étiquette provisoire, le lecteur ne voit rien pendant une minute."""
    import asyncio

    class _Instant:
        def _label(self, item):  # noqa: ANN001
            return {"t0": item["t0"], "t1": item["t1"], "speaker": "S0"}

    stage = GrowingBufferDiarStage(
        offline=_Scripted([[(0.0, 10.0, "a")]]),
        online=_Instant(),
        policy=GrowingBufferPolicy(first_s=1000.0, ratio=2.0),
    )
    out = asyncio.run(_drain(stage, [(0.0, 10.0), (12.0, 20.0)]))
    assert len(out) == 2, "chaque segment voisé doit recevoir une étiquette aussitôt"


def test_revisions_reach_the_callback_rather_than_being_lost():
    import asyncio

    seen: list = []
    stage = GrowingBufferDiarStage(
        offline=_Scripted(
            [
                [(0.0, 10.0, "a"), (10.0, 20.0, "a")],
                [(0.0, 10.0, "a"), (10.0, 20.0, "b")],
            ]
        ),
        policy=GrowingBufferPolicy(first_s=5.0, ratio=2.0),
        on_revision=seen.append,
    )
    asyncio.run(_drain(stage, [(0.0, 10.0), (10.0, 20.0)]))
    assert seen, "une passe qui démentit doit le signaler"


def test_an_empty_stream_closes_cleanly():
    import asyncio

    stage = GrowingBufferDiarStage(offline=_Scripted([[]]))
    assert asyncio.run(_drain(stage, [])) == []
