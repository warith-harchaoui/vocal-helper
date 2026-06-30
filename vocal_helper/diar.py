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
  as alternative). For inputs longer than ``ideal_duration_s`` (300 s
  for pyannote, 60 s for NeMo — values codified in the pdbms 2026-06-29
  ideal-duration sweep), the stage chunks with overlap, runs the
  backend per chunk and stitches via cosine AHC. This is the current
  best **offline** answer per pdbms §10.5 (AMI dev-slice median
  DER 0.116, inside Bredin 2023's band).

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
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from vocal_helper._settings import resolve_hf_token
from vocal_helper.types import DiarizedSegment, VoicedSegment

BackendName = Literal["pyannote", "nemo"]


@dataclass
class _Centroid:
    """One running speaker centroid."""

    speaker_id: str
    vector: NDArray[np.float32]   # L2-normalised
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
    hf_token : str, optional
        HuggingFace token, forwarded to the pyannote backend when
        the model isn't already cached. When ``None``, the value is
        resolved via :func:`vocal_helper._settings.resolve_hf_token`
        — ``$HF_TOKEN`` then ``secrets.hf_token`` in ``settings.yaml``.
    """

    def __init__(
        self,
        *,
        backend: BackendName = "pyannote",
        join_threshold: float = 0.30,
        ema_alpha: float = 0.1,
        min_segment_ms: int = 500,
        hf_token: str | None = None,
    ) -> None:
        if not 0.0 < join_threshold < 2.0:
            raise ValueError(f"join_threshold must be in (0, 2), got {join_threshold}")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        self.backend = backend
        self.join_threshold = join_threshold
        self.ema_alpha = ema_alpha
        self.min_segment_ms = min_segment_ms
        # Resolve the token eagerly so the cached value reflects the
        # state at construction time — calls into pyannote later cannot
        # be affected by a mid-run env / settings.yaml mutation.
        self.hf_token = resolve_hf_token(hf_token)
        self._embedder = None
        self._centroids: list[_Centroid] = []
        self._next_id = 0

    # ----- backend ------------------------------------------------------

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        if self.backend == "pyannote":
            self._embedder = _PyannoteEmbedder(hf_token=self.hf_token)
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
        """Consume :class:`VoicedSegment` from ``inbox``, push :class:`DiarizedSegment`."""
        self._ensure_embedder()
        while True:
            item = await inbox.get()
            if item is None:
                await outbox.put(None)
                return
            seg = self._label(item)
            if seg is not None:
                await outbox.put(seg)

    # ----- core ---------------------------------------------------------

    def _label(self, seg: VoicedSegment) -> DiarizedSegment | None:
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
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        if not self._centroids:
            return self._spawn(emb, t)
        # Cosine distance = 1 − cos_sim ; both unit-norm by construction.
        sims = np.array([float(emb @ c.vector) for c in self._centroids])
        dists = 1.0 - sims
        best = int(np.argmin(dists))
        if dists[best] <= self.join_threshold:
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
        sid = f"S{self._next_id}"
        self._next_id += 1
        self._centroids.append(_Centroid(
            speaker_id=sid,
            vector=emb.astype(np.float32, copy=False),
            n_updates=1,
            last_seen_t=t,
        ))
        return sid


# ---------------------------------------------------------------------------
# Backend wrappers — minimal, lazy.
# ---------------------------------------------------------------------------


class _PyannoteEmbedder:
    """Wraps ``pyannote.audio.Inference("pyannote/embedding")``."""

    def __init__(self, *, hf_token: str | None = None) -> None:
        self.hf_token = hf_token
        self._inference = None

    def load(self) -> None:
        try:
            from pyannote.audio import Inference, Model  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OnlineDiarStage(backend='pyannote') requires the pyannote extra. "
                "Install with `pip install vocal-helper[pyannote]`."
            ) from e
        model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=self.hf_token,
        )
        # Whole-segment embedding ; pyannote's Inference handles the
        # 160 ms minimum padding internally.
        self._inference = Inference(model, window="whole")

    def embed(self, pcm: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        import torch  # type: ignore

        if pcm.ndim != 1:
            raise ValueError(f"_PyannoteEmbedder expects mono PCM, got {pcm.shape}")
        wave = torch.from_numpy(pcm).unsqueeze(0)
        out = self._inference({"waveform": wave, "sample_rate": sr})
        return np.asarray(out, dtype=np.float32).reshape(-1)


class _TitaNetEmbedder:
    """Wraps NVIDIA TitaNet via NeMo for sharper cosine separation."""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OnlineDiarStage(backend='nemo') requires the nemo extra. "
                "Install with `pip install vocal-helper[nemo]`."
            ) from e
        self._model = EncDecSpeakerLabelModel.from_pretrained("titanet_large").eval()

    def embed(self, pcm: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        import torch  # type: ignore

        wave = torch.from_numpy(pcm).unsqueeze(0)
        length = torch.tensor([pcm.shape[0]], dtype=torch.long)
        with torch.no_grad():
            _, emb = self._model.forward(input_signal=wave, input_signal_length=length)
        return np.asarray(emb.squeeze(0).cpu().numpy(), dtype=np.float32)


# ===========================================================================
# OFFLINE PATH
# ===========================================================================


# Ideal duration constants — codified in the pdbms 2026-06-29
# ideal-duration sweep (``doc/studies/diar-study.md`` §10.6). For
# audio longer than this, the offline stage chunks + stitches ; for
# anything shorter it runs the backend as a single call.
IDEAL_DURATION_S_PYANNOTE = 300.0
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

    Parameters
    ----------
    backend : "pyannote" | "nemo"
        Backend to use. Default ``"pyannote"``.
    ideal_duration_s : float, optional
        Chunk length for the long-form path. Default depends on the
        backend (300 s for pyannote, 60 s for NeMo).
    overlap_s : float
        Overlap between adjacent chunks. Default 10 s.
    stitch_threshold : float
        Cosine-distance threshold for cross-chunk AHC stitching.
        Default 0.35.
    hf_token : str, optional
        HuggingFace token, forwarded to pyannote. When ``None``, the
        value is resolved via
        :func:`vocal_helper._settings.resolve_hf_token` — ``$HF_TOKEN``
        then ``secrets.hf_token`` in ``settings.yaml``.
    """

    def __init__(
        self,
        *,
        backend: BackendName = "pyannote",
        ideal_duration_s: float | None = None,
        overlap_s: float = 10.0,
        stitch_threshold: float = 0.35,
        hf_token: str | None = None,
    ) -> None:
        self.backend = backend
        if ideal_duration_s is None:
            ideal_duration_s = (
                IDEAL_DURATION_S_PYANNOTE if backend == "pyannote"
                else IDEAL_DURATION_S_NEMO
            )
        self.ideal_duration_s = ideal_duration_s
        self.overlap_s = overlap_s
        self.stitch_threshold = stitch_threshold
        self.hf_token = resolve_hf_token(hf_token)
        self._backend_obj: Any | None = None
        self._embedder: Any | None = None

    # ----- lifecycle ----------------------------------------------------

    def _ensure_backend(self) -> None:
        if self._backend_obj is not None:
            return
        if self.backend == "pyannote":
            self._backend_obj = _PyannoteOfflineDiar(hf_token=self.hf_token)
            self._embedder = _PyannoteEmbedder(hf_token=self.hf_token)
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
            await outbox.put(DiarizedSegment(
                t0=float(t0),
                t1=float(t1),
                sample_rate=sr,
                speaker=spk,
                pcm=pcm[i0:i1].copy(),
            ))
        await outbox.put(None)

    # ----- long-form chunking ------------------------------------------

    def _diarize_long(
        self,
        pcm: NDArray[np.float32],
        sr: int,
    ) -> list[tuple[float, float, str]]:
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
# Offline backend wrappers — minimal, lazy.
# ---------------------------------------------------------------------------


class _PyannoteOfflineDiar:
    """Wraps ``pyannote.audio.Pipeline('pyannote/speaker-diarization-3.1')``."""

    def __init__(self, *, hf_token: str | None = None) -> None:
        self.hf_token = hf_token
        self._pipeline = None

    def load(self) -> None:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OfflineDiarStage(backend='pyannote') requires the pyannote extra. "
                "Install with `pip install vocal-helper[pyannote]`."
            ) from e
        # pyannote.audio renamed the auth kwarg between major versions
        # — try the new name first, fall back.
        try:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", token=self.hf_token,
            )
        except TypeError:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token,
            )
        if self._pipeline is None:
            raise RuntimeError(
                "Failed to load pyannote/speaker-diarization-3.1 — visit the "
                "HF model page and accept the terms with the same account "
                "owning HF_TOKEN."
            )

    def diarize(
        self,
        pcm: NDArray[np.float32],
        sr: int,
    ) -> list[tuple[float, float, str]]:
        import torch  # type: ignore

        wave = torch.from_numpy(pcm).unsqueeze(0)
        ann = self._pipeline({"waveform": wave, "sample_rate": sr})
        return [
            (segment.start, segment.end, str(speaker))
            for segment, _track, speaker in ann.itertracks(yield_label=True)
        ]


class _NemoSortformerDiar:
    """Wraps NVIDIA ``nvidia/diar_sortformer_v1`` for batch use."""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OfflineDiarStage(backend='nemo') requires the nemo extra. "
                "Install with `pip install vocal-helper[nemo]`."
            ) from e
        self._model = SortformerEncLabelModel.from_pretrained(
            "nvidia/diar_sortformer_v1"
        ).eval()

    def diarize(
        self,
        pcm: NDArray[np.float32],
        sr: int,
    ) -> list[tuple[float, float, str]]:
        # Sortformer accepts a path or a tensor ; we use a per-call
        # temp WAV to keep the dependency surface small.
        import tempfile

        import soundfile as sf  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, pcm, sr, subtype="PCM_16")
            preds = self._model.diarize(audio=tmp.name, batch_size=1)
        # ``preds`` is a list of RTTM-like strings ; parse them.
        out: list[tuple[float, float, str]] = []
        for line in preds[0]:
            if not isinstance(line, str):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            t0 = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            out.append((t0, t0 + dur, str(spk)))
        return out
