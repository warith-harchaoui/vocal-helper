"""
Regression tests for :class:`vocal_helper.diar.OnlineDiarStage` batch repair.

Module summary
--------------
Reproduces — on a fully synthetic, model-free fixture — the online
diarizer's over-segmentation failure documented in
``DIARIZATION-TROUBLES.md`` (203 speaker labels for a 4-speaker file) and
pins the fix. The greedy single-pass clusterer mints a permanent new
speaker for every embedding that is farther than ``join_threshold`` from
all existing centroids, with no cap and no merge — so overlap / laughter /
jingle / backchannel outliers each spawn a throwaway singleton. The
``refine_on_close`` pass merges near-duplicate centroids and prunes those
micro-clusters into their nearest real speaker once the whole stream is
available.

The fixture uses a fake embedder returning pre-built vectors, so no
Whisper / pyannote / NeMo / torch is loaded — these run in the default
(non-integration) unit suite.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import asyncio
import collections

import numpy as np

from vocal_helper.diar import OnlineDiarStage
from vocal_helper.types import VoicedSegment

# Ground truth for the synthetic fixture.
N_TRUE_SPEAKERS = 4
SEGMENTS_PER_SPEAKER = 30
N_OUTLIERS = 30
EMB_DIM = 64
_SEG_DUR_S = 0.6  # comfortably above the 500 ms min_segment_ms floor


class _ScriptedEmbedder:
    """Fake embedder handing back a pre-built vector per :meth:`embed` call.

    ``OnlineDiarStage.run`` consumes segments in arrival order, so returning
    ``vectors[i]`` on the i-th call keeps embeddings aligned with segments
    without threading any signature through the PCM.
    """

    def __init__(self, vectors: list[np.ndarray]) -> None:
        self._vectors = vectors
        self._i = 0

    def load(self) -> None:  # pragma: no cover — trivial
        """No-op : the scripted vectors need no model."""

    def embed(self, pcm: np.ndarray, sr: int) -> np.ndarray:
        """Return the next scripted embedding, ignoring ``pcm`` / ``sr``."""
        v = self._vectors[self._i]
        self._i += 1
        return v


def _build_fixture() -> tuple[list[VoicedSegment], list[np.ndarray]]:
    """Build a 4-speaker fixture with outlier segments interleaved.

    Each real speaker is a tight cluster around one of four orthonormal
    basis vectors (inter-speaker cosine distance ~1.0, intra-speaker
    ~0.03). The outliers are independent random unit vectors, each far
    from every speaker and from each other — exactly the embeddings that
    make the greedy online clusterer mint singletons.

    Returns
    -------
    tuple[list[VoicedSegment], list[np.ndarray]]
        Segments in arrival order and the aligned embeddings for the
        scripted embedder.
    """
    rng = np.random.default_rng(20260716)
    # Four orthonormal speaker centroids via QR — pairwise cosine ~0.
    basis, _ = np.linalg.qr(rng.standard_normal((EMB_DIM, N_TRUE_SPEAKERS)))
    bases = [basis[:, k].astype(np.float32) for k in range(N_TRUE_SPEAKERS)]

    def _unit(v: np.ndarray) -> np.ndarray:
        return (v / np.linalg.norm(v)).astype(np.float32)

    tagged: list[np.ndarray] = []
    for spk in range(N_TRUE_SPEAKERS):
        for _ in range(SEGMENTS_PER_SPEAKER):
            tagged.append(_unit(bases[spk] + 0.03 * rng.standard_normal(EMB_DIM)))
    for _ in range(N_OUTLIERS):
        tagged.append(_unit(rng.standard_normal(EMB_DIM)))

    # Interleave so speakers and outliers are mixed through the timeline,
    # mimicking a real conversation rather than block-sorted speakers.
    order = rng.permutation(len(tagged))
    embeddings = [tagged[i] for i in order]

    segments: list[VoicedSegment] = []
    pcm = np.zeros(8, dtype=np.float32)  # unused by the fake embedder
    for i in range(len(embeddings)):
        t0 = i * _SEG_DUR_S
        segments.append(VoicedSegment(t0=t0, t1=t0 + _SEG_DUR_S, sample_rate=16_000, pcm=pcm))
    return segments, embeddings


def _run_stage(stage: OnlineDiarStage, segments: list[VoicedSegment]) -> list[str]:
    """Drive ``stage.run`` end-to-end and return the emitted speaker labels."""

    async def _drive() -> list[str]:
        inbox: asyncio.Queue = asyncio.Queue()
        outbox: asyncio.Queue = asyncio.Queue()
        for seg in segments:
            inbox.put_nowait(seg)
        inbox.put_nowait(None)
        await stage.run(inbox, outbox)
        labels: list[str] = []
        while True:
            item = outbox.get_nowait()
            if item is None:
                break
            labels.append(item["speaker"])
        return labels

    return asyncio.run(_drive())


def _distinct(labels: list[str]) -> set[str]:
    """Distinct real speaker labels, excluding the ``"S?"`` unknown bucket."""
    return {label for label in labels if label != "S?"}


def test_online_diar_over_segments_without_refine() -> None:
    """Baseline : the greedy online path explodes on outlier embeddings.

    This is the documented failure — every outlier mints its own permanent
    speaker, so the label count runs far past the four real speakers.
    """
    segments, embeddings = _build_fixture()
    stage = OnlineDiarStage(refine_on_close=False)
    stage._embedder = _ScriptedEmbedder(embeddings)

    labels = _run_stage(stage, segments)
    distinct = _distinct(labels)
    # Each of the ~30 outliers spawns its own throwaway speaker → far more
    # than N_TRUE_SPEAKERS. Guard loosely so the test tracks the mechanism,
    # not the exact count.
    assert len(distinct) >= N_TRUE_SPEAKERS + 15


def test_refine_on_close_recovers_true_speaker_count() -> None:
    """The refine pass collapses the explosion back to the real speakers.

    Definition of done (DIARIZATION-TROUBLES.md §7): distinct labels within
    a small margin of the true speaker count, and a low singleton ratio.
    """
    segments, embeddings = _build_fixture()
    stage = OnlineDiarStage(refine_on_close=True)
    stage._embedder = _ScriptedEmbedder(embeddings)

    labels = _run_stage(stage, segments)
    distinct = _distinct(labels)
    counts = collections.Counter(labels)

    # §7 pass criteria, adapted to the fixture's 4 true speakers.
    assert len(distinct) <= N_TRUE_SPEAKERS + 1
    singletons = sum(1 for label, c in counts.items() if label != "S?" and c <= 2)
    assert singletons / max(1, len(distinct)) < 0.15
    # The four dense clusters must survive intact — every real speaker's
    # segments should land on one label carrying ~30 segments.
    big = [c for label, c in counts.items() if label != "S?" and c >= SEGMENTS_PER_SPEAKER]
    assert len(big) == N_TRUE_SPEAKERS


def test_refine_labels_prunes_singleton_into_nearest_speaker() -> None:
    """Unit-level check of the prune step on a hand-built two-speaker set."""
    stage = OnlineDiarStage(refine_on_close=True, min_cluster_size=2)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    outlier = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # Two real speakers (3 segments each) + one lone outlier near speaker A.
    embeddings = [a, a, a, b, b, b, outlier]
    provisional = ["S0", "S0", "S0", "S1", "S1", "S1", "S2"]
    segs = [
        {"t0": 0.0, "t1": 0.6, "sample_rate": 16_000, "pcm": None, "speaker": p}
        for p in provisional
    ]
    # Nudge the outlier so its nearest survivor is unambiguously speaker A.
    embeddings[-1] = (0.6 * a + 0.4 * outlier).astype(np.float32)

    final = stage._refine_labels(segs, embeddings)
    # The outlier must be folded into a real speaker, leaving exactly two.
    assert len(set(final)) == 2
    # It sits closer to A, so it should share A's final label.
    assert final[-1] == final[0]


def test_max_speakers_cap_bounds_online_cluster_count() -> None:
    """``max_speakers`` forces out-of-threshold segments into existing ids."""
    segments, embeddings = _build_fixture()
    stage = OnlineDiarStage(refine_on_close=False, max_speakers=N_TRUE_SPEAKERS)
    stage._embedder = _ScriptedEmbedder(embeddings)

    labels = _run_stage(stage, segments)
    # Never more than the cap, regardless of how many outliers arrive.
    assert len(_distinct(labels)) <= N_TRUE_SPEAKERS
