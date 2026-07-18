"""
vocal_helper.diar
=================

Two diarization paths : **online** for live streams and **offline**
for batch / file-based inputs.

- :class:`OnlineDiarStage` — consumes :class:`VoicedSegment` as the
  VAD emits them, embeds each one, and runs a per-segment cosine
  running-mean clusterer. The current best **online** answer per
  the pdbms 2026-06-29 canonical study : matches ``hungarian_nemo``
  / ``hungarian_pyannote`` in spirit, simpler because the VAD has
  already isolated each segment.
- :class:`OfflineDiarStage` — receives the **full PCM buffer** and
  hands it to the canonical offline backend
  (``pyannote/speaker-diarization-3.1`` by default, NeMo Sortformer
  as alternative). Runs whole-buffer by default : the 2026-07-14
  offline map-reduce study found whole-buffer strictly best for DER,
  so pyannote only chunks past ``ideal_duration_s`` = 3600 s (a memory
  backstop), while NeMo keeps 60 s (Sortformer 90 s cap). When chunking
  does kick in, the stage overlaps chunks and stitches via cosine AHC
  (pdbms §10.5, AMI dev-slice median DER 0.116, inside Bredin 2023's band).

Reliability — which path to use
-------------------------------
The **offline** path is the reliable one and should be preferred for any
batch / file input. A 2026-07-16 DER sweep (``studies/diar_der_paths.py``,
pyannote.metrics, collar 0.25) measured every path against ground truth:

========================  ==================  =====================  =====================
corpus                    offline pyannote    offline nemo (Sortf)   online (no ref / ref)
========================  ==================  =====================  =====================
AMI (20-40 min meetings)  **0.122**           0.242                  0.497 / 0.351
bagarre (~30 s, <=4 spk)  0.338               **0.177**              0.586 / 0.592
========================  ==================  =====================  =====================

Takeaways: (1) offline pyannote is literature-grade (Bredin 2023 ~ 0.188
uncollared) and wins on long meetings, running whole-buffer with global
clustering and no speaker-count cap. (2) offline **NeMo Sortformer**
(``diar_sortformer_4spk-v1``) is end-to-end and overlap-aware and nearly
*halves* the DER on short ``<=4``-speaker clips — but it is capped at 4
speakers and ~90 s per window, so it degrades once it must chunk long audio.
(3) The **online** streaming path (nemo TitaNet embeddings) stays ~3x the
offline DER — a latency-bound approximation that cannot model overlapped
speech ; ``refine_on_close`` roughly halves its DER on meetings that
over-segment (ES2011a 0.588 -> 0.296) and never hurts.

Default policy: **pyannote** is the offline default — robust across any length
and speaker count, best on the long inputs that dominate ``file`` use. Pick
``--offline --diar-backend nemo`` for short ``<=4``-speaker workloads where
Sortformer wins. The CLI ``file --no-real-time`` auto-selects offline pyannote
when the bundle is present ; reserve :class:`OnlineDiarStage` for live streams.
Downstream integrators embedding diarization in a larger pipeline should use
:class:`OfflineDiarStage` / :class:`~vocal_helper.OfflinePipeline` for batch.

Online algorithm — minimal cosine-AHC online clusterer
------------------------------------------------------

Algorithm — minimal cosine-AHC online clusterer
-----------------------------------------------
We **don't** carry the full pdbms HungarianDiar across this
boundary. The full sliding-window Hungarian wrapper assumes the
diarizer is fed *raw PCM windows*, but here the VAD already gives
us isolated voiced segments — one embedding per segment is enough
and the global stitching collapses to a 1-D nearest-centroid match
on cosine distance with running-mean updates.

For each incoming :class:`VoicedSegment` :

1. Embed via the configured backend (pyannote/embedding or
   NVIDIA TitaNet). The embedding is L2-normalised.
2. Compute cosine distance to every existing centroid.
3. The minimum-distance centroid wins iff its distance is below
   ``join_threshold`` (default 0.30, calibrated on AMI dev-slice in
   the 2026-06-30 stitch-threshold sweep). Otherwise, mint a new
   speaker.
4. Update the matched centroid by exponential moving average with
   coefficient ``ema_alpha`` (default 0.1) so the centroid adapts
   slowly to within-speaker variation.

The stage is meant to run *online* — every voiced segment is
labelled at most ``embed_latency_ms`` after the speech ends.

Backend choice
--------------
``backend='pyannote'`` is the default and only requires the
``pyannote`` extra (``pip install vocal-helper[pyannote]``). It
uses ``pyannote/embedding`` (160 ms minimum input). ``backend='nemo'``
uses NVIDIA TitaNet via the ``nemo`` extra ; slower to load but
better cosine separation on noisy mixes.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from vocal_helper.types import DiarizedSegment, VoicedSegment

BackendName = Literal["pyannote", "nemo"]
DeviceName = Literal["cpu", "cuda", "mps"]


def _auto_torch_device(explicit: str | None) -> str:
    """Pick the torch device : explicit override, else CUDA > MPS > CPU.

    Pyannote 3.1 on CPU is ~ 10-20× real-time on Apple Silicon ;
    MPS gives roughly real-time. We auto-select rather than ask
    callers to remember the right knob.

    Returns
    -------
    str
        ``"cuda"``, ``"mps"`` or ``"cpu"``. Always a non-empty string
        so ``torch.device(returned)`` is always safe.
    """
    if explicit:
        return explicit
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class _Centroid:
    """One running speaker centroid."""

    speaker_id: str
    vector: NDArray[np.float32]  # L2-normalised
    n_updates: int = 0
    last_seen_t: float = field(default=0.0)


class OnlineDiarStage:
    """Producer/consumer online speaker diarizer.

    Parameters
    ----------
    backend : "pyannote" | "nemo"
        Which embedding model to use. Default ``"pyannote"``.
    join_threshold : float
        Cosine-distance threshold below which a new segment joins an
        existing centroid. Default 0.30 — calibrated on AMI dev-slice
        N=8 in 2026-06-30 stitch_threshold sweep, where the
        pyannote/embedding distribution exhibits a clear DER minimum
        at the 0.30-0.45 plateau.
    ema_alpha : float
        Exponential-moving-average coefficient for centroid updates.
        Default 0.1.
    min_segment_ms : int
        Minimum voiced-segment duration to attempt embedding.
        Default 500 ms (pyannote/embedding's convolutional kernels
        choke on shorter inputs).
    device : "cpu" | "cuda" | "mps", optional
        Torch device for the pyannote embedder. ``None`` (default)
        auto-picks CUDA > MPS > CPU. Has no effect on the NeMo backend.

    Notes
    -----
    Model weights load from the self-hosted diarization-engines bundle
    (see :func:`resolve_diarization_engines`) — no HuggingFace token is
    used or accepted.
    """

    def __init__(
        self,
        *,
        # Default ``"nemo"`` (TitaNet) selected by the 2026-06-30
        # embedding-backend sweep (``studies/diar_embedding_backend.py``)
        # on AMI dev-slice : TitaNet gives a 0.354 separability margin
        # (inter-speaker − intra-speaker median cosine distance) vs
        # 0.201 for pyannote/embedding — a 76 % uplift. The cost is
        # ~ 7 × per-call latency (45 ms vs 6 ms) which is negligible
        # in a streaming per-segment workload.
        # Fall back to ``"pyannote"`` if the NeMo install footprint is
        # prohibitive (NeMo + torch is ~ 5 GB ; pyannote alone is
        # ~ 500 MB). Pass ``backend="pyannote"`` explicitly to opt out.
        backend: BackendName = "nemo",
        join_threshold: float = 0.30,
        ema_alpha: float = 0.1,
        min_segment_ms: int = 500,
        device: str | None = None,
        max_speakers: int | None = None,
        refine_on_close: bool = False,
        min_cluster_size: int = 2,
        merge_threshold: float | None = None,
    ) -> None:
        """Configure the online diarizer ; the embedder loads lazily.

        Parameters
        ----------
        backend : "pyannote" | "nemo"
            Embedding backend. Default ``"nemo"`` (TitaNet) for its sharper
            cosine separation ; pass ``"pyannote"`` to opt out of the
            heavier NeMo install.
        join_threshold : float
            Cosine-distance threshold below which a segment joins an
            existing centroid. Default 0.30. Must be in ``(0, 2)``.
        ema_alpha : float
            Exponential-moving-average coefficient for centroid updates.
            Default 0.1. Must be in ``(0, 1]``.
        min_segment_ms : int
            Minimum voiced-segment duration to attempt embedding.
            Default 500 ms.
        device : str, optional
            Torch device for the pyannote embedder. ``None`` (default)
            auto-picks CUDA > MPS > CPU. No effect on the NeMo backend.
        max_speakers : int, optional
            Hard cap on the number of *online* speakers. Once this many
            centroids exist, an embedding that would otherwise mint a new
            speaker is forced into its nearest existing centroid instead.
            ``None`` (default) leaves the online path unbounded — the same
            behaviour as before this parameter existed. A cap is a blunt
            guard against runaway cluster proliferation on true live
            streams ; for batch input prefer ``refine_on_close``.
        refine_on_close : bool
            Batch-mode repair. When ``True``, the stage buffers every
            diarized segment (plus its embedding) and, once the stream
            closes, runs a global re-clustering pass — merging
            near-duplicate centroids (cosine distance ≤ ``merge_threshold``)
            and pruning micro-clusters smaller than ``min_cluster_size``
            into their nearest survivor — before emitting the whole batch
            with corrected, compact ``"S<n>"`` labels. This fixes the
            online clusterer's over-segmentation on long multi-speaker
            audio (see ``DIARIZATION-TROUBLES.md``) at the cost of holding
            the labelled segments until end-of-stream, so it is only for
            batch use where latency is already sacrificed. Default
            ``False`` (pure streaming, emit as you go).
        min_cluster_size : int
            Minimum number of segments a cluster must accumulate to survive
            the ``refine_on_close`` prune. Clusters below this are folded
            into their nearest surviving centroid. Default 2 (drop
            singletons). No effect unless ``refine_on_close`` is set.
        merge_threshold : float, optional
            Cosine-distance threshold for merging near-duplicate centroids
            during the ``refine_on_close`` pass. ``None`` (default) reuses
            ``join_threshold``. No effect unless ``refine_on_close`` is set.

        Raises
        ------
        ValueError
            If ``join_threshold`` is not in ``(0, 2)``, ``ema_alpha`` is not
            in ``(0, 1]``, ``max_speakers`` is set below 1, or
            ``min_cluster_size`` is below 1.
        """
        if not 0.0 < join_threshold < 2.0:
            raise ValueError(f"join_threshold must be in (0, 2), got {join_threshold}")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if max_speakers is not None and max_speakers < 1:
            raise ValueError(f"max_speakers must be >= 1 or None, got {max_speakers}")
        if min_cluster_size < 1:
            raise ValueError(f"min_cluster_size must be >= 1, got {min_cluster_size}")
        self.backend = backend
        self.join_threshold = join_threshold
        self.ema_alpha = ema_alpha
        self.min_segment_ms = min_segment_ms
        self.device = device
        self.max_speakers = max_speakers
        self.refine_on_close = refine_on_close
        self.min_cluster_size = min_cluster_size
        self.merge_threshold = merge_threshold if merge_threshold is not None else join_threshold
        self._embedder = None
        self._centroids: list[_Centroid] = []
        self._next_id = 0

    # ----- backend ------------------------------------------------------

    def _ensure_embedder(self) -> None:
        """Lazily instantiate and load the configured embedding backend.

        Idempotent — returns immediately once the embedder exists, so it
        is safe to call at the top of :meth:`run`.

        Raises
        ------
        ValueError
            If ``self.backend`` is neither ``"pyannote"`` nor ``"nemo"``.
        """
        if self._embedder is not None:
            return
        if self.backend == "pyannote":
            self._embedder = _PyannoteEmbedder(device=self.device)
        elif self.backend == "nemo":
            self._embedder = _TitaNetEmbedder()
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        self._embedder.load()

    # ----- public coroutine --------------------------------------------

    async def run(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
    ) -> None:
        """Consume :class:`VoicedSegment` from ``inbox``, push :class:`DiarizedSegment`.

        In streaming mode (``refine_on_close=False``) each segment is
        labelled and emitted as it arrives. In batch mode
        (``refine_on_close=True``) segments are buffered, globally
        re-clustered when the stream closes, then emitted with corrected
        labels — see :meth:`_run_refine`.
        """
        self._ensure_embedder()
        if self.refine_on_close:
            await self._run_refine(inbox, outbox)
            return
        while True:
            item = await inbox.get()
            if item is None:
                await outbox.put(None)
                return
            seg = self._label(item)
            if seg is not None:
                await outbox.put(seg)

    async def _run_refine(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
    ) -> None:
        """Batch path : buffer every segment, re-cluster globally on close.

        Runs the same greedy online assignment as the streaming path (so
        provisional labels and centroids build up identically), but holds
        each :class:`DiarizedSegment` and its embedding back instead of
        emitting. Once the upstream sends its ``None`` sentinel, a single
        global pass (:meth:`_refine_labels`) merges near-duplicate
        centroids and prunes micro-clusters ; the buffered segments are
        then emitted in arrival order with their corrected labels.
        """
        buffered: list[DiarizedSegment] = []
        embeddings: list[NDArray[np.float32] | None] = []
        while True:
            item = await inbox.get()
            if item is None:
                break
            seg, emb = self._label_capture(item)
            if seg is not None:
                buffered.append(seg)
                embeddings.append(emb)
        final_labels = self._refine_labels(buffered, embeddings)
        for seg, label in zip(buffered, final_labels, strict=True):
            seg["speaker"] = label
            await outbox.put(seg)
        await outbox.put(None)

    # ----- core ---------------------------------------------------------

    def _label(self, seg: VoicedSegment) -> DiarizedSegment | None:
        """Embed one voiced segment and assign it a speaker label.

        Segments shorter than ``min_segment_ms`` — or that raise inside the
        embedder — are labelled ``"S?"`` so callers can still ASR them
        without a confident speaker id.

        Parameters
        ----------
        seg : VoicedSegment
            The voiced segment to label, carrying its PCM and timing.

        Returns
        -------
        DiarizedSegment or None
            The segment tagged with a speaker id (a real ``"S<n>"`` label,
            or ``"S?"`` when embedding is skipped or fails).
        """
        sr = seg["sample_rate"]
        dur_ms = (seg["t1"] - seg["t0"]) * 1000.0
        if dur_ms < self.min_segment_ms:
            # Too short to embed reliably — assign to "S?" so callers
            # can still ASR it but without a confident speaker id.
            return DiarizedSegment(
                t0=seg["t0"],
                t1=seg["t1"],
                sample_rate=sr,
                speaker="S?",
                pcm=seg["pcm"],
            )
        try:
            emb = self._embedder.embed(seg["pcm"], sr)
        except Exception:  # noqa: BLE001 — embedder failure shouldn't kill the stream
            return DiarizedSegment(
                t0=seg["t0"],
                t1=seg["t1"],
                sample_rate=sr,
                speaker="S?",
                pcm=seg["pcm"],
            )
        speaker_id = self._assign(emb, t=seg["t1"])
        return DiarizedSegment(
            t0=seg["t0"],
            t1=seg["t1"],
            sample_rate=sr,
            speaker=speaker_id,
            pcm=seg["pcm"],
        )

    def _assign(self, emb: NDArray[np.float32], t: float) -> str:
        """Nearest-centroid match on cosine distance, else mint a speaker.

        L2-normalises ``emb``, then joins the closest existing centroid iff
        its cosine distance is ``<= join_threshold`` — updating that
        centroid by exponential moving average — otherwise spawns a new
        speaker via :meth:`_spawn`.

        Parameters
        ----------
        emb : NDArray[np.float32]
            The segment embedding (normalised in place).
        t : float
            End time of the segment in seconds ; recorded as the matched
            centroid's ``last_seen_t``.

        Returns
        -------
        str
            The speaker id (``"S<n>"``) the segment was assigned to.
        """
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        if not self._centroids:
            return self._spawn(emb, t)
        # Cosine distance = 1 − cos_sim ; both unit-norm by construction.
        sims = np.array([float(emb @ c.vector) for c in self._centroids])
        dists = 1.0 - sims
        best = int(np.argmin(dists))
        # Join the nearest centroid when it is close enough — or when the
        # ``max_speakers`` cap is reached, in which case an out-of-threshold
        # embedding is forced into its nearest existing speaker rather than
        # minting an unbounded new one.
        capped = self.max_speakers is not None and len(self._centroids) >= self.max_speakers
        if dists[best] <= self.join_threshold or capped:
            c = self._centroids[best]
            new_vec = (1.0 - self.ema_alpha) * c.vector + self.ema_alpha * emb
            n = float(np.linalg.norm(new_vec))
            if n > 0:
                new_vec /= n
            c.vector = new_vec.astype(np.float32, copy=False)
            c.n_updates += 1
            c.last_seen_t = t
            return c.speaker_id
        return self._spawn(emb, t)

    def _spawn(self, emb: NDArray[np.float32], t: float) -> str:
        """Mint a new speaker centroid seeded from ``emb``.

        Parameters
        ----------
        emb : NDArray[np.float32]
            The (already normalised) embedding to seed the new centroid.
        t : float
            End time of the segment in seconds, stored as ``last_seen_t``.

        Returns
        -------
        str
            The freshly-allocated speaker id (``"S<n>"``).
        """
        sid = f"S{self._next_id}"
        self._next_id += 1
        self._centroids.append(
            _Centroid(
                speaker_id=sid,
                vector=emb.astype(np.float32, copy=False),
                n_updates=1,
                last_seen_t=t,
            )
        )
        return sid

    # ----- batch refinement (refine_on_close) --------------------------

    def _label_capture(
        self, seg: VoicedSegment
    ) -> tuple[DiarizedSegment | None, NDArray[np.float32] | None]:
        """Label one segment *and* return its unit-norm embedding.

        Mirrors :meth:`_label` — same short-segment / embedder-failure
        fallbacks to ``"S?"`` — but additionally hands back the normalised
        embedding (or ``None`` when the segment was not embedded) so the
        batch path can re-cluster on the raw per-segment vectors rather than
        the drifting online centroids.

        Returns
        -------
        tuple[DiarizedSegment or None, NDArray or None]
            The provisionally-labelled segment and its embedding, or an
            ``"S?"`` segment with ``None`` when embedding was skipped/failed.
        """
        sr = seg["sample_rate"]
        dur_ms = (seg["t1"] - seg["t0"]) * 1000.0
        unknown = DiarizedSegment(
            t0=seg["t0"], t1=seg["t1"], sample_rate=sr, speaker="S?", pcm=seg["pcm"]
        )
        if dur_ms < self.min_segment_ms:
            return unknown, None
        try:
            emb = self._embedder.embed(seg["pcm"], sr)
        except Exception:  # noqa: BLE001 — embedder failure shouldn't kill the stream
            return unknown, None
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        speaker_id = self._assign(emb, t=seg["t1"])
        labelled = DiarizedSegment(
            t0=seg["t0"], t1=seg["t1"], sample_rate=sr, speaker=speaker_id, pcm=seg["pcm"]
        )
        return labelled, vec.astype(np.float32, copy=False)

    def _refine_labels(
        self,
        segs: list[DiarizedSegment],
        embeddings: list[NDArray[np.float32] | None],
    ) -> list[str]:
        """Global re-clustering + singleton prune over the buffered batch.

        Repairs the online clusterer's over-segmentation once the whole
        stream is available. Working from the per-provisional-speaker
        centroids (mean of that speaker's segment embeddings) :

        1. **Merge** provisional speakers whose centroids sit within
           ``merge_threshold`` cosine distance (single-linkage union-find) —
           this folds the duplicates that slow EMA centroid drift spawns for
           a speaker already modelled.
        2. **Prune** merged clusters smaller than ``min_cluster_size`` by
           reassigning each of their segments to the nearest *surviving*
           cluster centroid — this absorbs the outlier micro-speakers
           (overlap, laughter, jingle, backchannels) into a real speaker.
        3. **Relabel** survivors to compact ``"S0", "S1", …"`` ids ordered by
           first appearance.

        Segments that were never embedded keep their ``"S?"`` label.

        Parameters
        ----------
        segs : list[DiarizedSegment]
            The buffered segments, in arrival order.
        embeddings : list[NDArray or None]
            Unit-norm embedding per segment (``None`` for ``"S?"`` segments),
            index-aligned with ``segs``.

        Returns
        -------
        list[str]
            Final speaker id per segment, index-aligned with ``segs``.
        """
        # Group segment indices by their provisional online label, keeping
        # only embedded segments — "S?" segments pass through untouched.
        by_label: dict[str, list[int]] = {}
        for i, (seg, emb) in enumerate(zip(segs, embeddings, strict=True)):
            if emb is None:
                continue
            by_label.setdefault(seg["speaker"], []).append(i)
        if not by_label:
            return [seg["speaker"] for seg in segs]

        labels = list(by_label.keys())
        # Per-provisional centroid = unit-norm mean of its segment embeddings.
        centroids = np.stack(
            [_unit_mean([embeddings[i] for i in by_label[lab]]) for lab in labels], axis=0
        )

        # 1. Merge near-duplicate centroids via single-linkage union-find.
        parent = list(range(len(labels)))

        def find(x: int) -> int:
            """Return the union-find root of ``x`` with path compression.

            Parameters
            ----------
            x : int
                Index into ``parent`` whose set representative is wanted.

            Returns
            -------
            int
                The representative (root) index of the set containing ``x``.
            """
            # Walk up to the root, halving the path on the way so future
            # lookups on the same chain get shallower (path compression).
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            """Merge the sets containing ``a`` and ``b`` in place.

            Parameters
            ----------
            a : int
                First index to merge.
            b : int
                Second index to merge.

            Returns
            -------
            None
                ``parent`` is mutated in place; nothing is returned.
            """
            # Resolve both roots, then attach the higher-indexed root to the
            # lower one so labels stay deterministic across runs.
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        if len(labels) > 1:
            sim = centroids @ centroids.T
            dist = 1.0 - sim
            for a in range(len(labels)):
                for b in range(a + 1, len(labels)):
                    if dist[a, b] <= self.merge_threshold:
                        union(a, b)

        # Collect merged groups: root -> segment indices + recomputed centroid.
        group_members: dict[int, list[int]] = {}
        for li, lab in enumerate(labels):
            group_members.setdefault(find(li), []).extend(by_label[lab])
        group_roots = list(group_members.keys())
        group_centroid = {
            root: _unit_mean([embeddings[i] for i in members])
            for root, members in group_members.items()
        }

        # 2. Prune: a group survives iff it has >= min_cluster_size segments.
        survivors = [r for r in group_roots if len(group_members[r]) >= self.min_cluster_size]
        # Degenerate guard — if the prune would erase everything (every
        # cluster tiny), keep them all rather than emit nothing meaningful.
        if not survivors:
            survivors = group_roots

        # 3. Assign each segment a final group root: survivors keep theirs,
        # pruned groups route each segment to the nearest survivor centroid.
        survivor_mat = np.stack([group_centroid[r] for r in survivors], axis=0)
        final_root: dict[int, int] = {}
        for root, members in group_members.items():
            if root in survivors:
                for i in members:
                    final_root[i] = root
            else:
                for i in members:
                    emb = embeddings[i]
                    nearest = int(np.argmax(survivor_mat @ emb))
                    final_root[i] = survivors[nearest]

        # Compact, first-appearance-ordered relabelling of the survivors.
        compact: dict[int, str] = {}
        out: list[str] = []
        for i, seg in enumerate(segs):
            if i not in final_root:
                out.append(seg["speaker"])  # untouched "S?"
                continue
            root = final_root[i]
            if root not in compact:
                compact[root] = f"S{len(compact)}"
            out.append(compact[root])
        return out


def _unit_mean(vectors: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    """Return the L2-normalised mean of a list of vectors.

    Falls back to the raw mean when it is degenerate (zero norm), which
    only happens if the inputs cancel exactly — unit-norm embeddings make
    that vanishingly unlikely, but the guard keeps the result finite.
    """
    mean = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 0:
        mean = mean / norm
    return mean.astype(np.float32, copy=False)


def _parse_sortformer_segments(lines: Any) -> list[tuple[float, float, str]]:
    """Parse NeMo Sortformer's per-segment output into ``(t0, t1, speaker)``.

    Handles both formats the model can emit so the offline NeMo backend works
    across nemo-toolkit versions:

    - **Compact** ``"<start> <end> <speaker>"`` (e.g. ``"1.920 3.040 speaker_0"``)
      — what ``diar_sortformer_4spk-v1`` returns under nemo-toolkit 2.x. The
      previous parser only understood legacy RTTM and silently dropped every
      one of these lines, so the backend returned no speakers at all.
    - **Legacy RTTM** ``"SPEAKER <file> <chan> <start> <dur> <NA> <NA> <spk> …"``.

    Non-string / malformed / zero-length entries are skipped.

    Parameters
    ----------
    lines : Iterable
        The per-utterance prediction list Sortformer returns (``preds[0]``).

    Returns
    -------
    list[tuple[float, float, str]]
        ``[(t0, t1, speaker), …]`` in seconds.
    """
    out: list[tuple[float, float, str]] = []
    for line in lines:
        if not isinstance(line, str):
            continue
        parts = line.split()
        # Compact form: start, end, speaker.
        if len(parts) == 3:
            try:
                t0, t1 = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if t1 > t0:
                out.append((t0, t1, str(parts[2])))
            continue
        # Legacy RTTM: SPEAKER <file> <chan> <start> <dur> <NA> <NA> <spk> …
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            try:
                t0, dur = float(parts[3]), float(parts[4])
            except ValueError:
                continue
            out.append((t0, t0 + dur, str(parts[7])))
    return out


# ---------------------------------------------------------------------------
# Backend wrappers — minimal, lazy.
# ---------------------------------------------------------------------------


class _PyannoteEmbedder:
    """Wraps ``pyannote.audio.Inference("pyannote/embedding")``."""

    def __init__(self, *, device: str | None = None) -> None:
        """Store the requested device ; defer model loading to :meth:`load`.

        Parameters
        ----------
        device : str, optional
            Torch device for the embedder. ``None`` (default) auto-picks
            CUDA > MPS > CPU at load time.
        """
        self.device = device  # ``None`` → auto-pick at load time
        self._inference = None

    def load(self) -> None:
        """Build the ``pyannote/embedding`` inference from the local bundle.

        Loads the whole-segment ``pyannote/embedding`` checkpoint straight
        from the self-hosted diarization-engines bundle (no HuggingFace
        token, no network) and moves it to the chosen device, falling back
        to CPU if the requested backend can't run the forward path.

        Raises
        ------
        ImportError
            If the optional ``pyannote`` extra is not installed.
        RuntimeError
            If no ``pyannote/embedding`` weight is present in the
            diarization-engines bundle.
        """
        try:
            import torch  # type: ignore
            from pyannote.audio import Inference, Model  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OnlineDiarStage(backend='pyannote') requires the pyannote extra. "
                "Install with `pip install vocal-helper[pyannote]`."
            ) from e
        # Bundle-only : pyannote ``Model`` loads the local ``.bin`` checkpoint
        # directly, so no token and no network. There is no HF fallback.
        engines = resolve_diarization_engines()
        local_bin = (
            engines / "pyannote-embedding" / "pytorch_model.bin" if engines is not None else None
        )
        if local_bin is None or not local_bin.exists():
            raise RuntimeError(
                "No pyannote/embedding weight in the diarization-engines bundle. "
                "Set `engines.diarization_url` in settings.yaml (or "
                "$VH_DIARIZATION_ENGINES). No HuggingFace token is needed."
            )
        # Local checkpoint path — zero HuggingFace.
        model = Model.from_pretrained(str(local_bin))
        # Whole-segment embedding ; pyannote's Inference handles the
        # 160 ms minimum padding internally. ``device=`` makes Inference
        # move the model and incoming tensors to the right backend ;
        # not all pyannote ops support MPS yet, so we fall back to CPU
        # loudly if the forward path raises.
        chosen = _auto_torch_device(self.device)
        try:
            self._inference = Inference(
                model,
                window="whole",
                device=torch.device(chosen),
            )
        except (RuntimeError, NotImplementedError):
            self._inference = Inference(model, window="whole", device=torch.device("cpu"))

    def embed(self, pcm: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        """Return a single ``pyannote/embedding`` vector for one segment.

        Parameters
        ----------
        pcm : NDArray[np.float32]
            Mono PCM samples for the segment.
        sr : int
            Sample rate of ``pcm``.

        Returns
        -------
        NDArray[np.float32]
            The flattened, whole-segment speaker embedding.

        Raises
        ------
        ValueError
            If ``pcm`` is not 1-D (mono).
        """
        import torch  # type: ignore

        if pcm.ndim != 1:
            raise ValueError(f"_PyannoteEmbedder expects mono PCM, got {pcm.shape}")
        wave = torch.from_numpy(pcm).unsqueeze(0)
        out = self._inference({"waveform": wave, "sample_rate": sr})
        return np.asarray(out, dtype=np.float32).reshape(-1)


class _TitaNetEmbedder:
    """Wraps NVIDIA TitaNet via NeMo for sharper cosine separation."""

    def __init__(self) -> None:
        """Initialise with no model ; the checkpoint loads in :meth:`load`."""
        self._model = None

    def load(self) -> None:
        """Fetch the pretrained ``titanet_large`` model into eval mode.

        Raises
        ------
        ImportError
            If the optional ``nemo`` extra is not installed.
        """
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OnlineDiarStage(backend='nemo') requires the nemo extra. "
                "Install with `pip install vocal-helper[nemo]`."
            ) from e
        self._model = EncDecSpeakerLabelModel.from_pretrained("titanet_large").eval()

    def embed(self, pcm: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        """Return a single TitaNet speaker embedding for one segment.

        Parameters
        ----------
        pcm : NDArray[np.float32]
            Mono PCM samples for the segment.
        sr : int
            Sample rate of ``pcm`` (unused by TitaNet's forward path, kept
            for interface parity with :class:`_PyannoteEmbedder`).

        Returns
        -------
        NDArray[np.float32]
            The TitaNet speaker embedding vector.
        """
        import torch  # type: ignore

        wave = torch.from_numpy(pcm).unsqueeze(0)
        length = torch.tensor([pcm.shape[0]], dtype=torch.long)
        with torch.no_grad():
            _, emb = self._model.forward(input_signal=wave, input_signal_length=length)
        return np.asarray(emb.squeeze(0).cpu().numpy(), dtype=np.float32)


# ===========================================================================
# OFFLINE PATH
# ===========================================================================


# Ideal duration constants. For audio longer than this, the offline
# stage chunks + stitches ; for anything shorter it runs the backend as
# a single whole-buffer call.
#
# pyannote 3.1 handles long audio natively and the 2026-07-14 offline
# map-reduce study (``doc/studies/offline-mapreduce-study.md``) showed
# whole-buffer is strictly best for DER — chunk-and-stitch only *costs*
# quality (median DER 0.143 whole vs 0.170 at 300 s, cliffs below). So
# the pyannote default is set to run whole-buffer for any realistic
# meeting / podcast / lecture (≤ 1 h) and only falls back to chunking
# past that, purely as a memory backstop on extreme-length inputs.
IDEAL_DURATION_S_PYANNOTE = 3600.0
# NeMo Sortformer must chunk regardless : it degrades past its ~90 s
# training cap, so whole-buffer is not an option for that backend.
IDEAL_DURATION_S_NEMO = 60.0


class OfflineDiarStage:
    """Offline diarization on the full PCM buffer.

    Designed for batch / file-based use : the upstream source is
    expected to drain end-to-end, the stage collects the full PCM,
    then hands it to the canonical offline backend.

    Backends
    --------
    - ``"pyannote"`` — ``pyannote/speaker-diarization-3.1``. The
      production default for any meeting / podcast / lecture (pdbms
      §10.5 : AMI dev-slice median DER 0.116, inside Bredin 2023's
      0.188 band).
    - ``"nemo"`` — NVIDIA Sortformer (the ``nvidia/diar_sortformer_v1``
      checkpoint). Better for short clips ≤ 60 s, struggles past its
      90 s training cap. Auto-chunked when ``len > IDEAL_DURATION_S``.

    Long-form chunking
    ------------------
    For inputs longer than ``ideal_duration_s`` the stage replicates
    the pdbms ``ChunkedOfflineDiarizer`` strategy at minimal cost :

    1. Split the audio into chunks of ``ideal_duration_s`` with
       ``overlap_s`` (default 10 s) of shared content at boundaries.
    2. Run the backend on each chunk.
    3. Embed each chunk-local speaker on its concatenated audio.
    4. Cluster all chunk-local embeddings via cosine AHC
       (``stitch_threshold=0.35``, the value selected in the
       2026-06-30 stitch_threshold sweep on AMI dev-slice N=8 where
       t∈{0.30..0.40} forms the operating plateau).

    The full pdbms variant adds VAD-aware cut-point selection and
    pink-noise pad ; vocal-helper trades these for a simpler hard-cut
    + zero-pad pair to keep the dependency surface small. For mission-
    critical AMI-style work, use ``pdbms.diar.offline_chunked.ChunkedOfflineDiarizer``
    directly.

    Chunking is a memory ceiling, not a quality lever. The 2026-07-14
    offline map-reduce study (full stack VAD + ASR + diar on AMI,
    ``doc/studies/offline-mapreduce-study.md``) found DER strictly
    *monotone* in chunk size — whole-buffer is best (median DER 0.143 vs
    0.170 at 300 s, and cliffs to 0.31 / 0.50 at 120 s / 60 s as speaker
    fragmentation outruns the stitch) — and ASR *destabilises* when
    chunked (a long-window whisper loop drove one meeting to WER 1.17).
    So the **pyannote** default now runs whole-buffer for any realistic
    input (``ideal_duration_s`` = 3600 s) and only chunks past ~1 h as a
    memory backstop. **NeMo** is the exception: its Sortformer 90 s
    training cap forces chunking at ``ideal_duration_s`` = 60 s.

    Parameters
    ----------
    backend : "pyannote" | "nemo"
        Backend to use. Default ``"pyannote"``.
    ideal_duration_s : float, optional
        Whole-buffer ceiling : inputs longer than this are chunked +
        stitched, shorter ones run as a single call. Default depends on
        the backend — 3600 s for pyannote (effectively whole-buffer for
        any realistic meeting; chunking is a memory backstop only), 60 s
        for NeMo (forced by its Sortformer 90 s cap).
    overlap_s : float
        Overlap between adjacent chunks. Default 10 s.
    stitch_threshold : float
        Cosine-distance threshold for cross-chunk AHC stitching.
        Default 0.35.
    device : "cpu" | "cuda" | "mps", optional
        Torch device for the pyannote pipeline + embedder. ``None``
        (default) auto-picks CUDA > MPS > CPU. On Apple Silicon CPU
        is ~ 10× slower than MPS, so the auto-pick matters in practice.
        Has no effect on the NeMo backend.

    Notes
    -----
    Model weights load from the self-hosted diarization-engines bundle
    (see :func:`resolve_diarization_engines`) — no HuggingFace token is
    used or accepted.
    """

    def __init__(
        self,
        *,
        backend: BackendName = "pyannote",
        ideal_duration_s: float | None = None,
        overlap_s: float = 10.0,
        stitch_threshold: float = 0.35,
        device: str | None = None,
    ) -> None:
        """Configure the offline diarizer ; backends load lazily.

        Parameters
        ----------
        backend : "pyannote" | "nemo"
            Offline backend. Default ``"pyannote"``.
        ideal_duration_s : float, optional
            Whole-buffer ceiling : inputs longer than this are chunked +
            stitched, shorter ones run as a single call. ``None`` (default)
            picks the backend default — 3600 s for pyannote (a memory
            backstop), 60 s for NeMo (forced by its Sortformer 90 s cap).
        overlap_s : float
            Overlap between adjacent chunks when chunking. Default 10 s.
        stitch_threshold : float
            Cosine-distance threshold for cross-chunk AHC stitching.
            Default 0.35.
        device : str, optional
            Torch device for the pyannote pipeline + embedder. ``None``
            (default) auto-picks CUDA > MPS > CPU. No effect on NeMo.
        """
        self.backend = backend
        if ideal_duration_s is None:
            ideal_duration_s = (
                IDEAL_DURATION_S_PYANNOTE if backend == "pyannote" else IDEAL_DURATION_S_NEMO
            )
        self.ideal_duration_s = ideal_duration_s
        self.overlap_s = overlap_s
        self.stitch_threshold = stitch_threshold
        self.device = device
        self._backend_obj: Any | None = None
        self._embedder: Any | None = None

    # ----- lifecycle ----------------------------------------------------

    def _ensure_backend(self) -> None:
        """Lazily instantiate and load the diarizer plus its embedder.

        Idempotent — returns immediately once the backend exists. Pairs a
        whole-buffer diarizer with the matching embedder (used only for
        cross-chunk stitching on long inputs).

        Raises
        ------
        ValueError
            If ``self.backend`` is neither ``"pyannote"`` nor ``"nemo"``.
        """
        if self._backend_obj is not None:
            return
        if self.backend == "pyannote":
            self._backend_obj = _PyannoteOfflineDiar(device=self.device)
            self._embedder = _PyannoteEmbedder(device=self.device)
        elif self.backend == "nemo":
            self._backend_obj = _NemoSortformerDiar()
            self._embedder = _TitaNetEmbedder()
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        self._backend_obj.load()
        self._embedder.load()

    # ----- public API ---------------------------------------------------

    def diarize(
        self,
        pcm: NDArray[np.float32],
        sr: int,
    ) -> list[tuple[float, float, str]]:
        """Return ``[(t0, t1, speaker), …]`` sorted by start time."""
        self._ensure_backend()
        duration_s = pcm.shape[0] / float(sr)
        if duration_s <= self.ideal_duration_s + 1e-3:
            return self._backend_obj.diarize(pcm, sr)
        return self._diarize_long(pcm, sr)

    async def run(
        self,
        inbox: asyncio.Queue,
        outbox: asyncio.Queue,
    ) -> None:
        """Drain ``inbox`` of :class:`PcmFrame` ; emit :class:`DiarizedSegment`.

        Collects every frame until the upstream sends ``None``, then
        runs ``diarize`` on the full buffer in a worker thread and
        emits one :class:`DiarizedSegment` per identified speaker
        span.
        """
        self._ensure_backend()
        frames: list[NDArray[np.float32]] = []
        sr: int | None = None
        while True:
            item = await inbox.get()
            if item is None:
                break
            sr = item["sample_rate"]
            frames.append(item["pcm"])
        if not frames or sr is None:
            await outbox.put(None)
            return
        pcm = np.concatenate(frames, axis=0).astype(np.float32, copy=False)
        segs = await asyncio.to_thread(self.diarize, pcm, sr)
        # Emit one DiarizedSegment per (t0, t1, speaker), carrying the
        # corresponding PCM slice for the downstream ASR.
        for t0, t1, spk in segs:
            i0 = max(0, int(round(t0 * sr)))
            i1 = min(pcm.shape[0], int(round(t1 * sr)))
            if i1 <= i0:
                continue
            await outbox.put(
                DiarizedSegment(
                    t0=float(t0),
                    t1=float(t1),
                    sample_rate=sr,
                    speaker=spk,
                    pcm=pcm[i0:i1].copy(),
                )
            )
        await outbox.put(None)

    # ----- long-form chunking ------------------------------------------

    def _diarize_long(
        self,
        pcm: NDArray[np.float32],
        sr: int,
    ) -> list[tuple[float, float, str]]:
        """Chunk, diarize per chunk, then stitch speakers across chunks.

        Splits the buffer into ``ideal_duration_s`` windows with
        ``overlap_s`` shared content, diarizes each chunk, embeds each
        chunk-local speaker on its concatenated audio, then clusters all
        chunk-local embeddings via cosine AHC (``stitch_threshold``) to
        recover globally-consistent speaker ids. Neighbouring same-speaker
        spans that overlap from the chunk overlap region are merged.

        Parameters
        ----------
        pcm : NDArray[np.float32]
            The full mono PCM buffer.
        sr : int
            Sample rate of ``pcm``.

        Returns
        -------
        list[tuple[float, float, str]]
            ``[(t0, t1, speaker), …]`` in seconds, sorted by start time,
            with globally-stitched ``"S<n>"`` speaker ids.

        Raises
        ------
        ValueError
            If ``overlap_s`` is not strictly less than ``ideal_duration_s``.
        """
        ideal = int(self.ideal_duration_s * sr)
        overlap = int(self.overlap_s * sr)
        if overlap >= ideal:
            raise ValueError("overlap_s must be < ideal_duration_s")
        n = pcm.shape[0]
        chunk_segs: list[list[tuple[float, float, str, NDArray[np.float32]]]] = []
        cursor = 0
        while cursor < n:
            end = min(n, cursor + ideal)
            chunk = pcm[cursor:end]
            local = self._backend_obj.diarize(chunk, sr)
            # Build per-local-speaker embedding for cross-chunk stitching.
            by_label: dict[str, list[tuple[float, float]]] = {}
            for t0, t1, spk in local:
                by_label.setdefault(spk, []).append((t0, t1))
            this_chunk: list[tuple[float, float, str, NDArray[np.float32]]] = []
            cursor_s = cursor / float(sr)
            for spk, spans in by_label.items():
                pieces = []
                for t0, t1 in spans:
                    lo = max(0, int(round(t0 * sr)))
                    hi = min(chunk.shape[0], int(round(t1 * sr)))
                    if hi > lo:
                        pieces.append(chunk[lo:hi])
                if not pieces:
                    continue
                cat = np.concatenate(pieces, axis=0)
                if cat.shape[0] < sr // 2:
                    pad = np.zeros(sr // 2 - cat.shape[0], dtype=np.float32)
                    cat = np.concatenate([cat, pad], axis=0)
                try:
                    emb = self._embedder.embed(cat, sr)
                except Exception:  # noqa: BLE001
                    continue
                emb = np.asarray(emb, dtype=np.float32)
                nrm = float(np.linalg.norm(emb))
                if nrm > 0:
                    emb = emb / nrm
                for t0, t1 in spans:
                    this_chunk.append(
                        (cursor_s + t0, cursor_s + t1, spk, emb),
                    )
            chunk_segs.append(this_chunk)
            if end >= n:
                break
            cursor = max(cursor + 1, end - overlap)

        # Cross-chunk AHC stitching.
        all_emb_by_key: dict[tuple[int, str], NDArray[np.float32]] = {}
        all_segs: list[tuple[float, float, tuple[int, str]]] = []
        for ci, segs in enumerate(chunk_segs):
            for t0, t1, spk, emb in segs:
                key = (ci, spk)
                if key not in all_emb_by_key:
                    all_emb_by_key[key] = emb
                all_segs.append((t0, t1, key))
        if not all_emb_by_key:
            return []
        keys = list(all_emb_by_key.keys())
        embs = np.stack([all_emb_by_key[k] for k in keys], axis=0)
        if len(keys) == 1:
            gid_for = {keys[0]: 0}
        else:
            sim = embs @ embs.T
            dist = np.clip(1.0 - sim, 0.0, 2.0)
            from sklearn.cluster import AgglomerativeClustering  # type: ignore

            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    metric="precomputed",
                    linkage="average",
                    distance_threshold=self.stitch_threshold,
                )
            except TypeError:
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    affinity="precomputed",
                    linkage="average",
                    distance_threshold=self.stitch_threshold,
                )
            labels = clusterer.fit_predict(dist)
            gid_for = {k: int(lbl) for k, lbl in zip(keys, labels, strict=True)}

        # Emit globally-labelled segments, merging neighbouring
        # same-speaker spans that overlap from the chunk overlap region.
        labelled: list[tuple[float, float, str]] = [
            (t0, t1, f"S{gid_for[key]}") for t0, t1, key in all_segs
        ]
        labelled.sort(key=lambda x: x[0])
        merged: list[tuple[float, float, str]] = []
        for t0, t1, s in labelled:
            if merged and merged[-1][2] == s and t0 <= merged[-1][1] + 1e-3:
                merged[-1] = (merged[-1][0], max(merged[-1][1], t1), s)
            else:
                merged.append((t0, t1, s))
        return merged


# ---------------------------------------------------------------------------
# HF-free diarization engines — self-hosted weights, no HuggingFace at runtime.
# ---------------------------------------------------------------------------

# Self-hosted bundle of ALL model weights the project needs — the offline
# pyannote 3.1 pipeline, NeMo Sortformer, the online ``pyannote/embedding``
# embedder and SpeechBrain VoxLingua107. When present, every backend loads
# from it with zero HuggingFace access (no token, ``HF_HUB_OFFLINE=1`` safe).
# The canonical source is ``engines.diarization_url`` in ``settings.yaml`` ;
# this constant is only the last-resort default when nothing is configured.
DEFAULT_DIARIZATION_ENGINES_URL: str | None = "https://deraison.ai/diarization-engines.zip"


def resolve_diarization_engines() -> Path | None:
    """Locate the HF-free diarization-engines bundle, or ``None``.

    Source order: the explicit ``$VH_DIARIZATION_ENGINES`` env var, then
    ``engines.diarization_url`` in ``settings.yaml`` (the canonical
    config), then :data:`DEFAULT_DIARIZATION_ENGINES_URL`. A local dir is
    used as-is ; a URL to ``diarization-engines.zip`` is downloaded once
    and cached under ``$VH_CACHE_DIR`` (default ``~/.cache/vocal-helper``).
    Returns the directory that contains ``manifest.json``.
    """
    import os

    from vocal_helper._settings import resolve_diarization_engines_url

    # settings.yaml / env resolution first, then the built-in default.
    src = resolve_diarization_engines_url() or DEFAULT_DIARIZATION_ENGINES_URL
    if not src:
        return None

    if not src.startswith(("http://", "https://")):
        p = Path(src).expanduser()
        return p if (p / "manifest.json").exists() else (p if p.is_dir() else None)

    cache = Path(os.environ.get("VH_CACHE_DIR", Path.home() / ".cache" / "vocal-helper"))
    dest = cache / "diarization-engines"
    hits = list(dest.rglob("manifest.json")) if dest.exists() else []
    if hits:
        return hits[0].parent

    import tempfile
    import urllib.request
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        urllib.request.urlretrieve(src, tmp.name)
        with zipfile.ZipFile(tmp.name) as z:
            z.extractall(dest)
    hits = list(dest.rglob("manifest.json"))
    return hits[0].parent if hits else None


# ---------------------------------------------------------------------------
# Offline backend wrappers — minimal, lazy.
# ---------------------------------------------------------------------------


class _PyannoteOfflineDiar:
    """Wraps ``pyannote.audio.Pipeline('pyannote/speaker-diarization-3.1')``.

    Prefers the self-hosted :func:`resolve_diarization_engines` bundle
    (HF-free) ; falls back to the HuggingFace hub only when no bundle is
    configured.
    """

    def __init__(self, *, device: str | None = None) -> None:
        """Store the requested device ; defer pipeline loading to :meth:`load`.

        Parameters
        ----------
        device : str, optional
            Torch device for the pipeline. ``None`` (default) auto-picks
            CUDA > MPS > CPU at load time. The resolved device is recorded
            on ``self._device`` so :meth:`diarize` can place inputs
            correctly.
        """
        self.device = device  # ``None`` → auto-pick at load time
        self._pipeline = None
        # Resolved at load time so ``diarize`` knows where to put the
        # input tensor when it's not on the same device as the model.
        self._device: str = "cpu"

    def load(self) -> None:
        """Build the ``speaker-diarization-3.1`` pipeline from the local bundle.

        Loads the pipeline from the self-hosted diarization-engines
        bundle's local ``config.yaml`` (HF-free, no token, no network) and
        moves it to the chosen device, staying on CPU if the requested
        backend can't run every internal op.

        Raises
        ------
        ImportError
            If the optional ``pyannote`` extra is not installed.
        RuntimeError
            If no diarization-engines bundle is configured, or the local
            config fails to instantiate the pipeline.
        """
        try:
            import torch  # type: ignore
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OfflineDiarStage(backend='pyannote') requires the pyannote extra. "
                "Install with `pip install vocal-helper[pyannote]`."
            ) from e
        # Bundle-only : the self-hosted bundle carries a local ``config.yaml``
        # whose paths point at local ``.bin`` weights, so the pipeline never
        # touches HuggingFace. There is no HF fallback — a missing bundle is a
        # configuration error, not a reason to reach out to the hub.
        engines = resolve_diarization_engines()
        local_cfg = (
            engines / "pyannote-3.1" / "pyannote_diarization_config.yaml"
            if engines is not None
            else None
        )
        if local_cfg is None or not local_cfg.exists():
            raise RuntimeError(
                "No diarization-engines bundle found. Set `engines.diarization_url` "
                "in settings.yaml (or $VH_DIARIZATION_ENGINES) to the self-hosted "
                "diarization-engines bundle. No HuggingFace token is needed."
            )
        self._pipeline = self._load_local_pipeline(Pipeline, local_cfg)
        # A corrupt / incompatible local config would return None.
        if self._pipeline is None:
            raise RuntimeError("Failed to load the pyannote pipeline from the local bundle config.")
        # Move the pipeline to the right device. On Apple Silicon
        # CPU → MPS gives roughly 10× speed-up. Not all internal ops
        # support MPS yet ; on failure we keep the pipeline on CPU
        # rather than crash the whole stage.
        chosen = _auto_torch_device(self.device)
        if chosen != "cpu":
            try:
                self._pipeline.to(torch.device(chosen))
                self._device = chosen
            except (RuntimeError, NotImplementedError, AssertionError):
                # Stay on CPU — diarize will still work, just slower.
                self._device = "cpu"
        else:
            self._device = "cpu"

    def _load_local_pipeline(self, pipeline_cls: Any, config_path: Path) -> Any:
        """Load the pyannote pipeline from the local HF-free bundle.

        Parameters
        ----------
        pipeline_cls : Any
            The imported ``pyannote.audio.Pipeline`` class.
        config_path : Path
            Path to ``pyannote_diarization_config.yaml`` inside the
            bundle. Its ``embedding`` / ``segmentation`` entries are bare
            filenames resolved *relative to the config's own directory*.

        Returns
        -------
        Any
            The instantiated ``SpeakerDiarization`` pipeline.

        Notes
        -----
        pyannote resolves the weight paths against the process working
        directory, so we ``chdir`` into the config's directory for the
        duration of the call and restore the previous cwd afterwards.
        No token and no network are involved.
        """
        import os

        # Remember the caller's cwd so we can restore it no matter what.
        previous_cwd = os.getcwd()
        try:
            # The config references its weights by bare filename, so the
            # bundle directory must be the working directory at load time.
            os.chdir(config_path.parent)
            return pipeline_cls.from_pretrained(config_path.name)
        finally:
            # Always restore — a leaked cwd would corrupt every later
            # relative path in the host process.
            os.chdir(previous_cwd)

    def diarize(
        self,
        pcm: NDArray[np.float32],
        sr: int,
    ) -> list[tuple[float, float, str]]:
        """Run the pyannote 3.1 pipeline on a whole buffer.

        Places the input on the pipeline's device and unpacks the result,
        tolerating both the legacy bare ``Annotation`` return and the newer
        ``DiarizeOutput`` dataclass.

        Parameters
        ----------
        pcm : NDArray[np.float32]
            Mono PCM buffer to diarize.
        sr : int
            Sample rate of ``pcm``.

        Returns
        -------
        list[tuple[float, float, str]]
            ``[(t0, t1, speaker), …]`` in seconds, as emitted by pyannote.
        """
        import torch  # type: ignore

        # Match the input device to where the pipeline lives so MPS /
        # CUDA paths don't fall back to a silent CPU round-trip per
        # forward.
        wave = torch.from_numpy(pcm).unsqueeze(0).to(torch.device(self._device))
        out = self._pipeline({"waveform": wave, "sample_rate": sr})
        # pyannote 3.x changed its return type from a bare
        # ``pyannote.core.Annotation`` to a ``DiarizeOutput`` dataclass
        # exposing ``.speaker_diarization`` (the Annotation),
        # ``.speaker_embeddings`` and friends. Support both — the
        # check is one attribute access, not a version sniff.
        ann = getattr(out, "speaker_diarization", out)
        return [
            (segment.start, segment.end, str(speaker))
            for segment, _track, speaker in ann.itertracks(yield_label=True)
        ]


class _NemoSortformerDiar:
    """Wraps NVIDIA Sortformer (``diar_sortformer_4spk-v1``) for batch use.

    Prefers the self-hosted HF-free bundle's ``.nemo`` checkpoint
    (:func:`resolve_diarization_engines`) ; only falls back to the
    HuggingFace hub when no bundle is configured.

    Notes
    -----
    The upstream repo id is ``nvidia/diar_sortformer_4spk-v1`` — earlier
    code used ``nvidia/diar_sortformer_v1``, which 404s on HF. The bundle
    path sidesteps HF (and the id) entirely via ``restore_from``.
    """

    def __init__(self) -> None:
        """Initialise with no model ; the checkpoint restores in :meth:`load`."""
        # Lazily populated in ``load`` — kept ``None`` so import is cheap.
        self._model: Any | None = None

    def load(self) -> None:
        """Load the Sortformer model, preferring the local bundle.

        Raises
        ------
        ImportError
            If the optional ``nemo`` extra is not installed.
        """
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OfflineDiarStage(backend='nemo') requires the nemo extra. "
                "Install with `pip install vocal-helper[nemo]`."
            ) from e

        # Bundle-only : the ``.nemo`` checkpoint ships in the self-hosted
        # bundle and is restored locally — zero HuggingFace, no fallback.
        engines = resolve_diarization_engines()
        local_ckpt = None
        if engines is not None:
            # The builder ships exactly one ``.nemo`` under nemo-sortformer/.
            nemo_dir = engines / "nemo-sortformer"
            candidates = sorted(nemo_dir.glob("*.nemo")) if nemo_dir.exists() else []
            local_ckpt = candidates[0] if candidates else None

        if local_ckpt is None:
            raise RuntimeError(
                "No NeMo Sortformer checkpoint in the diarization-engines bundle. "
                "Set `engines.diarization_url` in settings.yaml (or "
                "$VH_DIARIZATION_ENGINES). No HuggingFace token is needed."
            )
        # Restore from the local file — no token, no network.
        self._model = SortformerEncLabelModel.restore_from(
            str(local_ckpt), map_location="cpu"
        ).eval()

    def diarize(
        self,
        pcm: NDArray[np.float32],
        sr: int,
    ) -> list[tuple[float, float, str]]:
        """Run Sortformer on a whole buffer via a temp WAV, parse its segments.

        Writes ``pcm`` to a per-call 16-bit temp WAV (keeping the
        dependency surface small), diarizes it, then parses the per-segment
        strings Sortformer returns — the compact ``"<start> <end> <speaker>"``
        form emitted by nemo-toolkit 2.x (and legacy RTTM lines) — into
        ``(t0, t1, speaker)`` tuples.

        Parameters
        ----------
        pcm : NDArray[np.float32]
            Mono PCM buffer to diarize.
        sr : int
            Sample rate of ``pcm``.

        Returns
        -------
        list[tuple[float, float, str]]
            ``[(t0, t1, speaker), …]`` in seconds, parsed from Sortformer's
            RTTM output.
        """
        # Sortformer accepts a path or a tensor ; we use a per-call
        # temp WAV to keep the dependency surface small.
        import tempfile

        import scipy.io.wavfile as _wav  # 16-bit PCM WAV, no soundfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
            _wav.write(tmp.name, sr, pcm16)
            preds = self._model.diarize(audio=tmp.name, batch_size=1)
        return _parse_sortformer_segments(preds[0])
