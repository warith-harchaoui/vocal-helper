"""
vocal_helper.cloud_media
========================

Preparing media for a paid call, and stitching the answer back together.

Three concerns that every hosted provider shares, kept in one place so neither
:mod:`vocal_helper.asr_cloud` nor :mod:`vocal_helper.diar_cloud` reimplements them:

1. **What gets uploaded.** Always a 16 kHz mono MP3 of the waveform, never the
   source container as it stands.
2. **How a file longer than a provider's cap is handled.** Overlapping chunks,
   then a stitch that keeps speaker labels globally consistent.
3. **Who spoke during an interval.** The overlap arithmetic both the stitch and
   the fusion need.

Why the upload is always converted, and always to the same thing
----------------------------------------------------------------
A hosted transcriber or diarizer only ever needs the waveform. Sending the source
file as it stands uploads a video track that will be discarded, or a 320 kbit/s
stereo master whose extra bits change no word. Converting unconditionally — not
merely above a size threshold — also makes the upload size and the processing time
predictable whatever the source happens to be, which matters when a run has
hundreds of files and a budget.

The conversion goes through ``video-helper`` for a video container and
``audio-helper`` otherwise, never through a hand-written ``ffmpeg`` command line:
those two libraries own the encoder flags for the whole suite, and a local copy of
them drifts.

Why chunks overlap, and why the seam is the overlap's middle
------------------------------------------------------------
Providers cap a single file (Gladia at ~135 minutes). Splitting at the cap and
concatenating gives two problems at every seam: a word cut in half, and two
independent sets of speaker labels with nothing relating them.

Overlapping the chunks solves both. The same voice speaks at the same absolute
time in both chunks, so the speaker who co-occurs most across the shared window is
the same person — that is what :func:`match_speakers_across_overlap` measures, and
it makes per-chunk labels globally consistent. Appending only the turns past the
**middle** of the overlap then dedupes the seam without losing anything: each
second of audio is contributed by exactly one chunk, and the one that contributes
it had context on both sides.

Usage example
-------------
>>> plan_chunks(duration_s=100.0, cap_s=8000.0, overlap_s=120.0)
[0.0]
>>> plan_chunks(duration_s=20000.0, cap_s=8000.0, overlap_s=120.0)
[0.0, 7880.0, 15760.0]
>>> speaker_by_overlap([{"start": 0.0, "end": 5.0, "speaker": "A"},
...                     {"start": 4.0, "end": 9.0, "speaker": "B"}], 4.5, 8.0)
'B'
"""

from __future__ import annotations

import collections
import math
from typing import Any

__all__ = [
    "audio_duration",
    "extract_clip",
    "match_speakers_across_overlap",
    "plan_chunks",
    "speaker_by_overlap",
    "upload_ready",
]


def upload_ready(audio: str, tmp_dir: str) -> str:
    """Return a 16 kHz mono MP3 of ``audio``, written inside ``tmp_dir``.

    Parameters
    ----------
    audio : str
        Source media: an audio file, or a video container whose video track is
        dropped.
    tmp_dir : str
        Directory the converted file is written into. **The caller owns its
        lifetime** — wrap the call in :func:`os_helper.temporary_folder` and
        cleanup happens on the way out, including on an exception, with no
        explicit delete here.

    Returns
    -------
    str
        Path of the converted file.
    """
    import os_helper as osh
    import video_helper as vh

    out = osh.join(tmp_dir, "upload.mp3")
    if vh.is_valid_video_file(audio):
        vh.extract_audio_track(audio, out, sample_rate=16000, channels=1, encoding="libmp3lame")
    else:
        import audio_helper as ah

        ah.sound_converter(audio, out, freq=16000, channels=1)
    return out


def audio_duration(path: str) -> float:
    """Duration of ``path`` in seconds, via ``audio-helper``.

    Parameters
    ----------
    path : str
        Media file.

    Returns
    -------
    float
        Duration in seconds.
    """
    import audio_helper as ah

    return float(ah.get_audio_duration(path))


def extract_clip(audio: str, start: float, length: float, out_path: str) -> str:
    """Extract ``[start, start + length]`` of ``audio`` into ``out_path``.

    Goes through ``audio-helper``'s chunk extraction rather than a hand-rolled
    ``ffmpeg`` call, and takes an explicit destination rather than minting its own
    temporary file: a long recording chunks into many clips, and a function that
    mints paths it does not own is how a process ends up leaking descriptors.

    Parameters
    ----------
    audio : str
        Source media.
    start : float
        Offset in seconds.
    length : float
        Clip length in seconds.
    out_path : str
        Destination, whose extension decides the container (use ``.mp3``).

    Returns
    -------
    str
        ``out_path``, for chaining.
    """
    import audio_helper as ah

    # Positional, and in this order: audio-helper's signature is
    # (audio_file, start_time, end_time, output_audio_filename). Naming the first
    # two as keywords while passing the destination positionally silently sent the
    # output path where the start time belongs — a defect this call had until it was
    # checked against the real signature rather than remembered.
    ah.extract_audio_chunk(audio, start, start + length, out_path)
    return out_path


def plan_chunks(*, duration_s: float, cap_s: float, overlap_s: float) -> list[float]:
    """Offsets of the overlapping chunks covering ``duration_s``.

    Parameters
    ----------
    duration_s : float, keyword-only
        Total media duration in seconds.
    cap_s : float, keyword-only
        The provider's per-file cap, in seconds. Stay a little under the published
        value: a cap hit is a rejected job, not a truncated one.
    overlap_s : float, keyword-only
        How much consecutive chunks share. It must be long enough for several
        speaker turns to occur inside it, or the label matching has nothing to
        measure.

    Returns
    -------
    list of float
        Chunk start offsets. A single ``[0.0]`` when the media fits in one call.

    Examples
    --------
    >>> plan_chunks(duration_s=60.0, cap_s=8000.0, overlap_s=120.0)
    [0.0]
    >>> plan_chunks(duration_s=16000.0, cap_s=8000.0, overlap_s=120.0)
    [0.0, 7880.0, 15760.0]

    The last chunk is deliberately allowed to be short: three chunks of 7880 s of
    fresh audio each are needed to reach 16000 s, and the third one carries only
    240 s. Two would stop at 15760 s and lose the tail.
    """
    if duration_s <= cap_s:
        return [0.0]
    step = cap_s - overlap_s
    count = math.ceil((duration_s - overlap_s) / step)
    return [i * step for i in range(count)]


def speaker_by_overlap(segments: list[dict[str, Any]], t0: float, t1: float) -> str | None:
    """The speaker whose segments overlap ``[t0, t1]`` most, or ``None``.

    Scans **every** segment rather than bisecting around ``t0``'s sorted position.
    A hosted diarizer returns genuinely overlapping segments on cross-talk, so a
    long segment starting well before ``t0`` can still cover it; a "start near
    ``t0``" shortcut silently drops exactly those.

    Parameters
    ----------
    segments : list of dict
        Each with ``start``, ``end`` and ``speaker``. Order is irrelevant.
    t0, t1 : float
        The interval to attribute.

    Returns
    -------
    str or None
        The dominant speaker label, or ``None`` when nothing overlaps.

    Examples
    --------
    >>> segs = [{"start": 0.0, "end": 10.0, "speaker": "A"},
    ...         {"start": 2.0, "end": 3.0, "speaker": "B"}]
    >>> speaker_by_overlap(segs, 2.0, 3.0)
    'A'
    >>> speaker_by_overlap(segs, 50.0, 60.0) is None
    True
    """
    overlap: collections.Counter = collections.Counter()
    for seg in segments:
        s0, s1 = float(seg["start"]), float(seg["end"])
        if s1 <= t0 or s0 >= t1:
            continue
        shared = min(t1, s1) - max(t0, s0)
        if shared > 0:
            overlap[seg["speaker"]] += shared
    return overlap.most_common(1)[0][0] if overlap else None


def match_speakers_across_overlap(
    base: list[dict[str, Any]],
    new: list[dict[str, Any]],
    win_start: float,
    win_end: float,
) -> dict[str, str]:
    """Map each ``new``-chunk speaker onto the ``base`` speaker it co-occurs with.

    Both chunks transcribed the same audio inside ``[win_start, win_end]``, so one
    voice speaks at the same absolute time in both. The ``new`` label whose turns
    overlap a ``base`` label's turns most is the same person.

    Parameters
    ----------
    base : list of dict
        Turns accumulated so far, in absolute time, each with ``start``, ``end``
        and ``speaker``.
    new : list of dict
        The next chunk's turns, already shifted to absolute time.
    win_start, win_end : float
        The shared window.

    Returns
    -------
    dict of str
        ``{new_label: base_label}``, containing only labels that were observed in
        the window. A caller must give an unmatched label a chunk-scoped name of
        its own rather than guessing.

    Examples
    --------
    >>> base = [{"start": 0.0, "end": 4.0, "speaker": "S0"}]
    >>> new = [{"start": 0.5, "end": 3.5, "speaker": "S1"}]
    >>> match_speakers_across_overlap(base, new, 0.0, 4.0)
    {'S1': 'S0'}
    """
    in_base = [u for u in base if u["end"] > win_start and u["start"] < win_end]
    in_new = [u for u in new if u["end"] > win_start and u["start"] < win_end]
    shared: collections.defaultdict = collections.defaultdict(collections.Counter)
    for n in in_new:
        for b in in_base:
            overlap = min(n["end"], b["end"]) - max(n["start"], b["start"])
            if overlap > 0:
                shared[n["speaker"]][b["speaker"]] += overlap
    return {label: counts.most_common(1)[0][0] for label, counts in shared.items() if counts}
