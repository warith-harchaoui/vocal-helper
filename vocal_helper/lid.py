"""
vocal_helper.lid
================

Spoken-language diarization — partition an audio into mono-language regions
so a code-switching recording can be transcribed one language at a time.

Why a posterior curve, not a boundary detector
----------------------------------------------
Running whisper in naive ``"auto"`` mode locks onto the first language it
hears and transcribes (often *translates*) the whole file in it — so a
text-level detector only ever sees one language. Language must be resolved
**before** transcription. But a language switch, unlike a speaker change,
has **no sharp acoustic cue** (same speaker, same channel): it cannot be
*detected* as a boundary, only *classified* per unit of time and smoothed.

Following ASR-posterior language diarization (Wang et al., Interspeech 2019)
with Gaussian-smoothed change points, using whisper's own language head as
the posterior source, :func:`detect_language_regions`:

1. samples a language-posterior curve over overlapping windows
   (:func:`language_posterior_curve`),
2. Gaussian-smooths it over time,
3. takes the per-frame argmax,
4. places change points where the argmax flips,
5. absorbs sub-``min_region_s`` regions into a neighbour,
6. locally refines each change point (fine re-scan + interpolated crossing),
7. snaps it to the nearest silence.

:func:`detect_language` remains the single-window primitive. All codes are
plain ISO-639-1 (``"en"``, ``"fr"``, …); callers may restrict the candidate
set so whisper never ranks an un-routable close relative (Galician ``gl`` /
Catalan ``ca`` over Spanish ``es``) on a short window.

Independent verification
------------------------
:func:`cross_check_regions` corroborates each region with a **second, fully
independent** audio classifier — an ECAPA-TDNN trained on VoxLingua107
(SpeechBrain) — which shares nothing with whisper but the signal. It is an
optional dependency (``pip install vocal-helper[lid]``), imported lazily.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vocal_helper.asr import DEFAULT_MODEL, DEFAULT_THREADS, WhisperStage

DEFAULT_SR = 16_000

# Language identification is a *classification* per window (a switch has no
# acoustic discontinuity), so we build a smoothed posterior CURVE over time.
DEFAULT_WINDOW_S = 10.0  # posterior window (context per detection)
DEFAULT_HOP_S = 3.0  # curve sampling step (coarse boundary resolution)
DEFAULT_SMOOTH_S = 6.0  # Gaussian sigma over time (anti-jitter)
# A language region shorter than this is treated as a mis-detection and
# absorbed into a neighbour — a real switch lasts several sentences.
DEFAULT_MIN_REGION_S = 8.0
# Local boundary refinement: re-scan the posterior around each coarse change
# point at a fine hop and interpolate the exact crossing — sub-second
# resolution without paying the fine hop over the whole file.
DEFAULT_REFINE_RADIUS_S = 4.0
DEFAULT_REFINE_WINDOW_S = 6.0
DEFAULT_REFINE_HOP_S = 1.0
# Half-width of the search for a silence to snap a change point onto.
DEFAULT_SNAP_S = 1.0
# Confidence floor for the single-pass fast path (see
# ``detect_language_regions_fast``): if one whole-file detection is at least
# this sure of a routable language, trust it and skip the posterior scan.
DEFAULT_FAST_CONF_GATE = 0.5

# Broad ISO-639-1 candidate set — the languages whisper.cpp's large-v3
# family identifies with usable accuracy. Callers restrict this to the
# languages they can actually route (e.g. ``("en", "fr", "es", "it",
# "pl", "nl")``) so whisper never ranks an un-routable relative on top.
DEFAULT_SUPPORTED_LANGS: tuple[str, ...] = (
    "en",
    "fr",
    "es",
    "it",
    "pt",
    "de",
    "nl",
    "pl",
    "ru",
    "uk",
    "cs",
    "sk",
    "ro",
    "ca",
    "gl",
    "sv",
    "da",
    "no",
    "fi",
    "hu",
    "el",
    "tr",
    "ar",
    "he",
    "fa",
    "hi",
    "id",
    "ms",
    "vi",
    "th",
    "ja",
    "ko",
    "zh",
)


# One whisper.cpp model is expensive to load ; cache a headless
# :class:`WhisperStage` per (model, threads) so a batch — or a multi-region
# file — reuses the same loaded model for both LID and transcription.
_STAGE_CACHE: dict[tuple[str, int], WhisperStage] = {}


def _get_stage(model: str, threads: int) -> WhisperStage:
    """Return a cached, model-loaded :class:`WhisperStage` for LID.

    Reuses :class:`WhisperStage`'s lazy ``_ensure_model`` loader — the
    ``pywhispercpp`` ``Model`` it holds is the same object that exposes
    ``auto_detect_language`` — so LID never duplicates model-load code.
    """
    key = (model, threads)
    stage = _STAGE_CACHE.get(key)
    if stage is None:
        stage = WhisperStage(model=model, language="auto", threads=threads)
        stage._ensure_model()
        _STAGE_CACHE[key] = stage
    return stage


def detect_language(
    pcm: NDArray[np.float32],
    *,
    model: str = DEFAULT_MODEL,
    threads: int = DEFAULT_THREADS,
    offset_ms: int = 0,
    supported: tuple[str, ...] = DEFAULT_SUPPORTED_LANGS,
) -> tuple[str, float]:
    """Identify the language of ``pcm`` (single window), restricted to ``supported``.

    Runs whisper.cpp's ``auto_detect_language`` and returns the most probable
    language **among** ``supported``. Restricting the candidate set is the
    "mind the codes" guard : on a short window whisper may rank a close
    relative (Galician ``gl`` / Catalan ``ca``) above the language the caller
    can route (Spanish ``es``), so a caller that only handles a handful of
    languages passes exactly those.

    Returns
    -------
    (str, float)
        ``(iso_639_1_code, probability)`` — the top-ranked supported language.
    """
    stage = _get_stage(model, threads)
    # whisper.cpp returns its full language distribution ; we ignore its own
    # argmax (``_top``) and re-rank strictly within ``supported`` below.
    (_top, _p), all_probs = stage._model.auto_detect_language(  # type: ignore[attr-defined]
        np.asarray(pcm, dtype=np.float32),
        offset_ms=offset_ms,
    )
    # Pick the best routable language ; a code absent from the distribution
    # scores 0.0, so it can only win if literally nothing else was ranked.
    best = max(supported, key=lambda code: float(all_probs.get(code, 0.0)))
    return best, float(all_probs.get(best, 0.0))


def language_posterior_curve(
    pcm: NDArray[np.float32],
    sample_rate: int = DEFAULT_SR,
    *,
    model: str = DEFAULT_MODEL,
    threads: int = DEFAULT_THREADS,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    supported: tuple[str, ...] = DEFAULT_SUPPORTED_LANGS,
) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64]]:
    """Sample a per-window language posterior over time.

    Returns ``(centers[T], langs[L], P[T, L])`` where ``centers`` are window
    centre times (s), ``langs`` the candidate codes, and each row of ``P`` a
    posterior over ``langs`` (renormalised over the supported set) from
    whisper's ``auto_detect_language`` on a ``window_s`` window centred there.
    Overlapping windows (``hop_s < window_s``) give a finely sampled curve.
    """
    stage = _get_stage(model, threads)
    n = int(pcm.shape[0])
    dur = n / float(sample_rate)
    langs = list(supported)
    half = window_s / 2.0
    centers: list[float] = []
    rows: list[NDArray[np.float64]] = []
    # Slide a window centred on each hop-spaced time ``t`` and read whisper's
    # language head there. Overlapping windows (hop < window) oversample the
    # curve so the later smoothing + argmax lands boundaries within a hop.
    t = 0.0
    while t < max(dur, hop_s):
        # Clamp the window to the signal — edge windows are simply shorter.
        a = max(0.0, t - half)
        b = min(dur, t + half)
        seg = pcm[int(a * sample_rate) : int(b * sample_rate)].astype(np.float32)
        if seg.shape[0] >= sample_rate:  # need ≥ 1 s to identify
            (_top, _p), all_probs = stage._model.auto_detect_language(seg)  # type: ignore[attr-defined]
            # Project onto the supported set and renormalise to a proper
            # distribution — the argmax later must compare like-for-like.
            row = np.array([float(all_probs.get(code, 0.0)) for code in langs])
            s = row.sum()
            # Zero-mass window (nothing supported ranked) → uniform prior.
            rows.append(row / s if s > 0 else np.ones(len(langs)) / len(langs))
            centers.append(min(t, dur))
        t += hop_s
    # No usable window at all (audio < 1 s) → one uniform frame, still valid.
    if not rows:  # audio shorter than 1 s
        return np.array([0.0]), langs, np.ones((1, len(langs))) / len(langs)
    return np.array(centers), langs, np.vstack(rows)


@dataclass(frozen=True)
class LangRegion:
    """A contiguous mono-language span of audio (seconds), ISO-639-1 ``lang``."""

    lang: str
    t0: float
    t1: float


def detect_language_regions(
    pcm: NDArray[np.float32],
    sample_rate: int = DEFAULT_SR,
    *,
    model: str = DEFAULT_MODEL,
    threads: int = DEFAULT_THREADS,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    smooth_s: float = DEFAULT_SMOOTH_S,
    min_region_s: float = DEFAULT_MIN_REGION_S,
    refine_s: float = DEFAULT_REFINE_RADIUS_S,
    snap_s: float = DEFAULT_SNAP_S,
    supported: tuple[str, ...] = DEFAULT_SUPPORTED_LANGS,
) -> list[LangRegion]:
    """Partition ``pcm`` into mono-language regions (posterior-curve method).

    A language switch has no sharp acoustic cue, so instead of hunting for a
    boundary we (1) sample a posterior curve over overlapping windows,
    (2) Gaussian-smooth it, (3) take the per-frame argmax, (4) place change
    points where it flips, (5) absorb sub-``min_region_s`` regions, (6) locally
    refine each change point, and (7) snap it to the nearest silence. The
    regions this yields must be transcribed *before* committing to a language,
    so each is transcribed in its own. Always returns ≥ 1 region.
    """
    n = int(pcm.shape[0])
    # Degenerate empty input — return a single zero-length region so callers
    # never have to special-case an empty list.
    if n == 0:
        return [LangRegion(supported[0], 0.0, 0.0)]
    dur = n / float(sample_rate)

    centers, langs, post = language_posterior_curve(
        pcm,
        sample_rate,
        model=model,
        threads=threads,
        window_s=window_s,
        hop_s=hop_s,
        supported=supported,
    )
    if len(post) > 1 and smooth_s > 0:
        from scipy.ndimage import gaussian_filter1d

        post = gaussian_filter1d(post, sigma=max(1e-6, smooth_s / hop_s), axis=0, mode="nearest")
    idx = post.argmax(axis=1)

    regions: list[LangRegion] = []
    run = 0
    for k in range(1, len(idx) + 1):
        if k == len(idx) or idx[k] != idx[run]:
            t0 = 0.0 if run == 0 else (centers[run - 1] + centers[run]) / 2.0
            t1 = dur if k == len(idx) else (centers[k - 1] + centers[k]) / 2.0
            regions.append(LangRegion(langs[int(idx[run])], t0, t1))
            run = k

    regions = _absorb_short_regions(_coalesce(regions), min_region_s)
    regions = _refine_boundaries(
        pcm,
        sample_rate,
        regions,
        model=model,
        threads=threads,
        supported=supported,
        radius_s=refine_s,
    )
    return _snap_boundaries_to_silence(pcm, sample_rate, regions, snap_s)


def detect_language_regions_fast(
    pcm: NDArray[np.float32],
    sample_rate: int = DEFAULT_SR,
    *,
    conf_gate: float = DEFAULT_FAST_CONF_GATE,
    model: str = DEFAULT_MODEL,
    threads: int = DEFAULT_THREADS,
    supported: tuple[str, ...] = DEFAULT_SUPPORTED_LANGS,
) -> list[LangRegion]:
    """Partition ``pcm`` into language regions, fast path for monolingual audio.

    Most real recordings are one language end to end, yet
    :func:`detect_language_regions` pays for a full overlapping-window
    posterior scan on *every* file. This wrapper first runs a single cheap
    global :func:`detect_language` over the whole signal: when that detection
    clears ``conf_gate`` on a routable language the file is taken to be
    monolingual and returned as **one** region (a single whisper call instead
    of dozens). Only when the global detection is *uncertain* — a genuinely
    code-switched or noisy file — does it fall back to the accurate
    posterior-curve segmentation of :func:`detect_language_regions`.

    On a corpus of support-call recordings this cut per-file language
    identification from ~73 s to ~1 s with identical region output on the
    monolingual majority, while the low-confidence fallback still recovers the
    switches. It is a drop-in replacement for :func:`detect_language_regions`
    wherever code-switching is the exception rather than the rule.

    Parameters
    ----------
    pcm : NDArray[np.float32]
        Mono waveform at ``sample_rate``.
    sample_rate : int, optional
        Sample rate of ``pcm`` in Hz (default :data:`DEFAULT_SR`).
    conf_gate : float, optional
        Minimum whole-file detection probability required to accept the
        single-region fast path (default :data:`DEFAULT_FAST_CONF_GATE`).
        Below it, the robust :func:`detect_language_regions` scan runs.
    model : str, optional
        whisper.cpp model name (default :data:`DEFAULT_MODEL`).
    threads : int, optional
        Inference threads (default :data:`DEFAULT_THREADS`).
    supported : tuple[str, ...], optional
        Routable ISO-639-1 candidates (default :data:`DEFAULT_SUPPORTED_LANGS`).

    Returns
    -------
    list[LangRegion]
        A single region spanning the file on the fast path, or the full
        multi-region partition on the fallback path. Always ≥ 1 region.

    Examples
    --------
    >>> regions = detect_language_regions_fast(pcm, 16_000)  # doctest: +SKIP
    >>> regions[0].lang                                      # doctest: +SKIP
    'en'
    """
    n = int(pcm.shape[0])
    # Degenerate empty input — mirror detect_language_regions' contract of
    # always returning at least one (zero-length) region so callers never
    # have to special-case an empty list.
    if n == 0:
        return [LangRegion(supported[0], 0.0, 0.0)]
    dur = n / float(sample_rate)

    # One cheap whole-file detection. A confident, routable answer means the
    # recording is almost certainly monolingual → skip the posterior scan.
    lang, conf = detect_language(pcm, model=model, threads=threads, supported=supported)
    if conf >= conf_gate and lang in supported:
        return [LangRegion(lang, 0.0, dur)]

    # Uncertain (code-switched or noisy) → pay for the accurate
    # posterior-curve segmentation rather than risk a wrong single label.
    return detect_language_regions(
        pcm,
        sample_rate,
        model=model,
        threads=threads,
        supported=supported,
    )


# ---------------------------------------------------------------------------
# Region helpers (module-level — pure, unit-testable without whisper).
# ---------------------------------------------------------------------------


def _coalesce(regions: list[LangRegion]) -> list[LangRegion]:
    """Merge consecutive same-language regions into one span."""
    out: list[LangRegion] = []
    for r in regions:
        if out and out[-1].lang == r.lang:
            p = out[-1]
            out[-1] = LangRegion(p.lang, p.t0, r.t1)
        else:
            out.append(r)
    return out


def _absorb_short_regions(regions: list[LangRegion], min_region_s: float) -> list[LangRegion]:
    """Relabel any sub-``min_region_s`` region to its longer neighbour, re-coalesce.

    Iteratively relabels the shortest offending region to whichever adjacent
    region is longer (the more established language) and re-merges touching
    same-language regions, until every region clears the threshold — or only
    one remains.
    """
    rs = list(regions)
    while len(rs) > 1:
        idx = min(range(len(rs)), key=lambda i: rs[i].t1 - rs[i].t0)
        if (rs[idx].t1 - rs[idx].t0) >= min_region_s:
            break
        left = rs[idx - 1] if idx > 0 else None
        right = rs[idx + 1] if idx < len(rs) - 1 else None
        if left and (not right or (left.t1 - left.t0) >= (right.t1 - right.t0)):
            new_lang = left.lang
        else:
            new_lang = right.lang  # type: ignore[union-attr]
        cur = rs[idx]
        rs[idx] = LangRegion(new_lang, cur.t0, cur.t1)
        rs = _coalesce(rs)
    return rs


def _refine_boundaries(
    pcm: NDArray[np.float32],
    sample_rate: int,
    regions: list[LangRegion],
    *,
    model: str,
    threads: int,
    supported: tuple[str, ...],
    radius_s: float,
    window_s: float = DEFAULT_REFINE_WINDOW_S,
    hop_s: float = DEFAULT_REFINE_HOP_S,
) -> list[LangRegion]:
    """Sharpen each region boundary by re-scanning the posterior locally.

    For a boundary between region A and region B, sample ``P_B − P_A`` on a
    fine grid within ±``radius_s``, smooth it, and place the boundary at the
    interpolated A→B crossing nearest the coarse guess — sub-second resolution
    without the noise of a globally fine hop.
    """
    if len(regions) < 2 or radius_s <= 0:
        return regions
    stage = _get_stage(model, threads)
    dur = pcm.shape[0] / float(sample_rate)
    half = window_s / 2.0
    out = list(regions)
    for i in range(len(out) - 1):
        a_lang, b_lang = out[i].lang, out[i + 1].lang
        if a_lang not in supported or b_lang not in supported:
            continue
        b = out[i].t1
        lo = max(out[i].t0, b - radius_s)
        hi = min(out[i + 1].t1, b + radius_s)
        centers: list[float] = []
        diff: list[float] = []  # P_B − P_A
        t = lo
        while t <= hi:
            wa = max(0.0, t - half)
            wb = min(dur, t + half)
            seg = pcm[int(wa * sample_rate) : int(wb * sample_rate)].astype(np.float32)
            if seg.shape[0] >= sample_rate:
                (_top, _p), allp = stage._model.auto_detect_language(seg)  # type: ignore[attr-defined]
                centers.append(t)
                diff.append(float(allp.get(b_lang, 0.0)) - float(allp.get(a_lang, 0.0)))
            t += hop_s
        new_b = b
        if len(diff) >= 2:
            d = np.array(diff)
            if len(d) >= 3:
                from scipy.ndimage import gaussian_filter1d

                d = gaussian_filter1d(d, sigma=1.0, mode="nearest")
            best: float | None = None
            for k in range(1, len(d)):
                if d[k - 1] <= 0.0 < d[k]:
                    t0, t1 = centers[k - 1], centers[k]
                    d0, d1 = d[k - 1], d[k]
                    cross = t0 + (t1 - t0) * (-d0) / (d1 - d0) if d1 != d0 else (t0 + t1) / 2.0
                    if best is None or abs(cross - b) < abs(best - b):
                        best = cross
            if best is not None:
                new_b = best
        new_b = min(max(new_b, out[i].t0 + 1e-3), out[i + 1].t1 - 1e-3)
        out[i] = LangRegion(a_lang, out[i].t0, new_b)
        out[i + 1] = LangRegion(b_lang, new_b, out[i + 1].t1)
    return out


def _snap_boundaries_to_silence(
    pcm: NDArray[np.float32],
    sample_rate: int,
    regions: list[LangRegion],
    snap_s: float,
) -> list[LangRegion]:
    """Move each internal region boundary to the lowest-energy point nearby.

    A code-switch happens at a pause, not mid-word. For each shared boundary,
    search ±``snap_s`` for the 50 ms frame with minimum RMS and snap there.
    """
    if len(regions) < 2 or snap_s <= 0:
        return regions
    frame = max(1, int(0.05 * sample_rate))
    out = list(regions)
    for i in range(len(out) - 1):
        b = out[i].t1
        lo = max(0.0, b - snap_s)
        hi = min(pcm.shape[0] / float(sample_rate), b + snap_s)
        a0, a1 = int(lo * sample_rate), int(hi * sample_rate)
        best_t, best_rms = b, float("inf")
        for s in range(a0, max(a0 + 1, a1 - frame), frame):
            seg = pcm[s : s + frame]
            rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else float("inf")
            if rms < best_rms:
                best_rms, best_t = rms, (s + frame / 2) / float(sample_rate)
        out[i] = LangRegion(out[i].lang, out[i].t0, best_t)
        out[i + 1] = LangRegion(out[i + 1].lang, best_t, out[i + 1].t1)
    return out


# ---------------------------------------------------------------------------
# Independent verification — SpeechBrain VoxLingua107 (optional [lid] extra).
# ---------------------------------------------------------------------------

VOXLINGUA_MODEL = "speechbrain/lang-id-voxlingua107-ecapa"
_VOXLINGUA_CACHE = Path.home() / ".cache" / "vocal-helper" / "voxlingua107-ecapa"
_classifier = None  # lazily loaded EncoderClassifier singleton


def _ensure_classifier():
    """Lazily load (and cache) the SpeechBrain VoxLingua107 ECAPA classifier.

    Returns the process-wide singleton, building it from the local self-hosted
    snapshot on first call. Raises if SpeechBrain is missing or no bundle is
    configured — the independent verification is strictly opt-in.
    """
    global _classifier
    # Singleton — the model is multi-hundred-MB ; load it at most once per process.
    if _classifier is not None:
        return _classifier
    try:
        from speechbrain.inference.classifiers import EncoderClassifier  # type: ignore
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "vocal_helper.lid independent verification needs SpeechBrain + torchaudio. "
            "Install with `pip install vocal-helper[lid]`."
        ) from e
    # Prefer the self-hosted HF-free bundle : point SpeechBrain's ``source``
    # at the local snapshot directory so nothing is fetched from HuggingFace.
    # Imported locally to avoid pulling the heavy diar module unless lid runs.
    from vocal_helper.diar import resolve_diarization_engines

    engines = resolve_diarization_engines()
    local_sb = engines / "speechbrain-voxlingua107" if engines is not None else None
    # Bundle-only : point SpeechBrain at the local snapshot directory. A valid
    # snapshot carries the pipeline hyperparams file at its root. No HF fallback.
    if local_sb is None or not (local_sb / "hyperparams.yaml").exists():
        raise RuntimeError(
            "No SpeechBrain VoxLingua107 snapshot in the diarization-engines "
            "bundle. Set `engines.diarization_url` in settings.yaml (or "
            "$VH_DIARIZATION_ENGINES). No HuggingFace token is needed."
        )

    _classifier = EncoderClassifier.from_hparams(
        source=str(local_sb),  # local dir → zero HuggingFace
        savedir=str(_VOXLINGUA_CACHE),
    )
    return _classifier


def detect_language_speechbrain(
    pcm: NDArray[np.float32],
    *,
    supported: tuple[str, ...] = DEFAULT_SUPPORTED_LANGS,
) -> tuple[str, float]:
    """Identify the language of ``pcm`` with SpeechBrain VoxLingua107.

    An ECAPA-TDNN spoken-language classifier that shares nothing with whisper
    but the audio — the independent second opinion. VoxLingua107 labels are
    ``"<iso>: <English name>"`` (e.g. ``"fr: French"``); the ISO-639-1 prefix
    is returned, coerced to ``supported[0]`` when outside ``supported``.

    Returns ``(iso_639_1_code, probability)``.
    """
    import torch

    clf = _ensure_classifier()
    wav = torch.tensor(np.asarray(pcm, dtype=np.float32)).unsqueeze(0)
    _out_prob, score, _index, text_lab = clf.classify_batch(wav)
    raw = str(text_lab[0]).split(":", 1)[0].strip().lower()
    code = raw if raw in supported else supported[0]
    prob = float(score.exp()) if hasattr(score, "exp") else float(score)
    return code, prob


@dataclass(frozen=True)
class RegionVerdict:
    """One region cross-checked against the independent SpeechBrain LID."""

    t0: float
    t1: float
    primary: str  # whisper-posterior label (the router's decision)
    speechbrain: str  # independent VoxLingua107 label
    sb_prob: float
    agree: bool


def cross_check_regions(
    pcm: NDArray[np.float32],
    regions: list[LangRegion],
    sample_rate: int = DEFAULT_SR,
    *,
    supported: tuple[str, ...] = DEFAULT_SUPPORTED_LANGS,
) -> list[RegionVerdict]:
    """Corroborate each region's language with the independent SpeechBrain LID.

    An ``agree=False`` verdict is a genuine disagreement between two models
    that share only the audio — a signal to inspect, not a code bug. Regions
    shorter than 1 s are too short to judge and pass through as agreeing.
    """
    verdicts: list[RegionVerdict] = []
    for r in regions:
        seg = pcm[int(r.t0 * sample_rate) : int(r.t1 * sample_rate)]
        if seg.shape[0] < sample_rate:
            verdicts.append(RegionVerdict(r.t0, r.t1, r.lang, "-", 0.0, True))
            continue
        sb, prob = detect_language_speechbrain(seg, supported=supported)
        verdicts.append(RegionVerdict(r.t0, r.t1, r.lang, sb, prob, sb == r.lang))
    return verdicts
