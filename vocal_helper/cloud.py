"""
vocal_helper.cloud
==================

The two **paid** entry points, siblings of the two free ones.

vocal-helper has four modes on two independent axes. *Local or cloud* asks where
the audio goes; *offline or live* asks when the answer arrives. Confusing the two
is the classic mistake, so they are named separately and none of them is reachable
by accident:

====================  ==================================  ============================
mode                  class                               engines
====================  ==================================  ============================
local, free, offline  :class:`vocal_helper.OfflinePipeline`   whisper.cpp + pyannote / NeMo / sherpa
local, free, live     :class:`vocal_helper.Pipeline`          whisper.cpp + NeMo TitaNet
paid, offline         :class:`CloudOfflinePipeline`           Gladia ``solaria-3`` + pyannoteAI
paid, live            :class:`CloudLivePipeline`              Gladia ``solaria-1`` + pyannoteAI Live-1
====================  ==================================  ============================

Four names rather than one factory taking a string, deliberately: a billed mode
must not be reachable through a typo, and the two local names already have live
callers that must keep working untouched.

There is **no automatic failover** between the columns. A missing key makes a paid
mode raise, never degrade: a run that silently fell back would produce a different
answer at a different quality under the same command, which is the worst possible
outcome for anything meant to be reproducible.

The hybrid is not a class
-------------------------
Measured on this suite's material, the free path does not lose on the words, it
collapses on the speakers. Diarization is also the cheaper hosted stage. So the
purchase worth making is **local transcription with hosted diarization**, and on
this corpus it changes the bill by roughly a factor of six.

That mode needs no new orchestration, because :class:`OfflinePipeline` already
composes source → diarization → transcription. It is one extra diarization
backend::

    voh.OfflinePipeline(source=..., config=voh.OfflinePipelineConfig(
        diar={"backend": "pyannoteai"},   # hosted speakers
        asr={"language": "fr"},           # local words
    ))

Expressing it that way rather than as a third class is what keeps the matrix
honest: the hybrid is genuinely the local offline pipeline with one stage bought,
not a fourth topology.

Usage example
-------------
>>> config = CloudOfflineConfig(transcriber="gladia", languages=["fr"])
>>> config.transcriber, config.diarizer
('gladia', 'pyannoteai')
>>> CloudOfflineConfig(transcriber="whisper").transcriber
Traceback (most recent call last):
    ...
ValueError: unknown transcriber 'whisper'; choose from elevenlabs, gladia
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from vocal_helper.asr_cloud import (
    GladiaLiveASR,
    elevenlabs_transcribe,
    gladia_utterances,
)
from vocal_helper.asr_cloud import pcm_f32_to_wire_bytes as _asr_wire
from vocal_helper.diar_cloud import PyannoteAiLiveDiar, pyannoteai_diarize
from vocal_helper.diar_cloud import pcm_f32_to_wire_bytes as _diar_wire
from vocal_helper.fusion import fuse_live_transcripts, relabel_utterances
from vocal_helper.types import CloudUtterance

__all__ = [
    "TRANSCRIBERS",
    "CloudLivePipeline",
    "CloudOfflineConfig",
    "CloudOfflinePipeline",
]

TRANSCRIBERS = ("gladia", "elevenlabs")
"""The hosted transcribers. ``gladia`` is the default; ``elevenlabs`` is selectable."""


@dataclass
class CloudOfflineConfig:
    """What a paid offline run should do.

    Parameters
    ----------
    transcriber : str, optional
        ``"gladia"`` (default, ``solaria-3``) or ``"elevenlabs"`` (Scribe).
    diarizer : str, optional
        ``"pyannoteai"``, or ``""`` to keep the transcriber's own speakers. Gladia
        diarizes its pre-recorded output, and Scribe bundles diarization too, but
        on a crowded panel the hosted diarizer is what the money is for — so it is
        on by default.
    languages : list of str, optional
        Expected languages. ``None`` lets the provider detect.
    code_switching : bool, optional
        Whether one recording may switch between the declared languages.
    glossary : list of str, optional
        Domain terms and proper nouns to bias toward. Never part of the billing
        ledger key, so editing it and re-running the same audio costs nothing.
    num_speakers, min_speakers, max_speakers : int, optional
        Diarization hints. **Prefer ``min_speakers``**: a roster is a floor, not a
        cap, and an exact count will merge or split to hit the number even when the
        audio disagrees.
    overlap_s : float, optional
        Shared audio between chunks when the media exceeds the provider's cap.

    Raises
    ------
    ValueError
        On an unknown transcriber or diarizer, at construction time rather than
        halfway through a billed run.

    Examples
    --------
    >>> CloudOfflineConfig().transcriber
    'gladia'
    >>> CloudOfflineConfig(diarizer="").diarizer
    ''
    >>> CloudOfflineConfig(diarizer="whisper")
    Traceback (most recent call last):
        ...
    ValueError: unknown diarizer 'whisper'; choose from '' or pyannoteai
    """

    transcriber: str = "gladia"
    diarizer: str = "pyannoteai"
    languages: list[str] | None = None
    code_switching: bool = True
    glossary: list[str] | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    overlap_s: float = 120.0

    def __post_init__(self) -> None:
        if self.transcriber not in TRANSCRIBERS:
            raise ValueError(
                f"unknown transcriber {self.transcriber!r}; "
                f"choose from {', '.join(sorted(TRANSCRIBERS))}"
            )
        if self.diarizer not in ("", "pyannoteai"):
            raise ValueError(
                f"unknown diarizer {self.diarizer!r}; choose from '' or pyannoteai"
            )


def _from_gladia(utterances: list[dict], engine: str) -> list[CloudUtterance]:
    """Convert Gladia pre-recorded utterances into the shared output type.

    Gladia returns whole utterances with word-level timings, so there is nothing to
    group — only to rename onto this library's vocabulary.
    """
    out: list[CloudUtterance] = []
    for utterance in utterances:
        words = utterance.get("words") or []
        out.append(
            {
                "t0": float(utterance["start"]),
                "t1": float(utterance["end"]),
                "speaker": str(utterance.get("speaker", "?")),
                "text": str(utterance.get("text", "")).strip(),
                "words": [
                    (float(w["start"]), float(w["end"]), str(w["word"])) for w in words
                ],
                "language": utterance.get("language"),
                "word_confidence": [float(w.get("confidence", 1.0)) for w in words],
                "engine": engine,
            }
        )
    return out


def _scribe_confidence(word: dict) -> float:
    """Turn Scribe's per-word log-probability into a confidence in ``[0, 1]``.

    Scribe reports ``logprob``, a natural logarithm, so the probability is its
    exponential. A missing field reads as ``1.0`` rather than ``0.0``: an absent
    measurement is not evidence of a bad word, and treating it as one would send a
    downstream repair pass after every word Scribe happened not to score.

    Parameters
    ----------
    word : dict
        One Scribe word object.

    Returns
    -------
    float
        Confidence, clamped to ``[0, 1]``.

    Examples
    --------
    >>> _scribe_confidence({"logprob": 0.0})
    1.0
    >>> round(_scribe_confidence({"logprob": -2.0}), 4)
    0.1353
    >>> _scribe_confidence({})
    1.0
    """
    import math

    if "logprob" not in word:
        return 1.0
    try:
        return max(0.0, min(1.0, math.exp(float(word["logprob"]))))
    except (TypeError, ValueError, OverflowError):
        return 1.0


def _from_elevenlabs(
    payload: dict,
    engine: str,
    *,
    max_gap_s: float = 0.8,
) -> list[CloudUtterance]:
    """Group Scribe's flat word stream into utterances.

    Scribe returns one long list of words rather than turns, so a boundary has to
    be decided here. Two rules, both conservative:

    a **speaker change** always ends an utterance, and a silence longer than
    ``max_gap_s`` within one speaker ends it too. Nothing is merged across a
    speaker change, because a merged turn attributes somebody's words to somebody
    else — the one error in a transcript that cannot be repaired by reading it.
    """
    words = [w for w in (payload.get("words") or []) if str(w.get("text", "")).strip()]
    out: list[CloudUtterance] = []
    current: list[dict] = []

    def flush() -> None:
        if not current:
            return
        out.append(
            {
                "t0": float(current[0]["start"]),
                "t1": float(current[-1]["end"]),
                "speaker": str(current[0].get("speaker_id", "?")),
                "text": "".join(str(w["text"]) for w in current).strip(),
                "words": [
                    (float(w["start"]), float(w["end"]), str(w["text"]).strip()) for w in current
                ],
                "language": payload.get("language_code"),
                "word_confidence": [_scribe_confidence(w) for w in current],
                "engine": engine,
            }
        )
        current.clear()

    for word in words:
        if current:
            same_speaker = word.get("speaker_id") == current[-1].get("speaker_id")
            gap = float(word["start"]) - float(current[-1]["end"])
            if not same_speaker or gap > max_gap_s:
                flush()
        current.append(word)
    flush()
    return out


class CloudOfflinePipeline:
    """Paid offline transcription and diarization: a file in, utterances out.

    Unlike the local pipelines, there is no PCM stream to drive: the providers take
    a whole file, so this class is a plain call rather than an async generator. The
    output type is the same either way, so a consumer does not branch on the mode.

    Parameters
    ----------
    config : CloudOfflineConfig, optional
        Defaults to Gladia ``solaria-3`` plus pyannoteAI, which is the combination
        this suite measured and bought.

    Examples
    --------
    >>> pipeline = CloudOfflinePipeline()
    >>> pipeline.config.transcriber, pipeline.config.diarizer
    ('gladia', 'pyannoteai')
    >>> pipeline.engine_name()
    'gladia:solaria-3+pyannoteai'
    >>> CloudOfflinePipeline(CloudOfflineConfig(diarizer="")).engine_name()
    'gladia:solaria-3'
    """

    def __init__(self, config: CloudOfflineConfig | None = None) -> None:
        self.config = config or CloudOfflineConfig()

    def engine_name(self) -> str:
        """Name the engines that produced a result, for the record it ends up in.

        Returns
        -------
        str
            e.g. ``"gladia:solaria-3+pyannoteai"``. A record names what wrote it:
            without that, a result from a degraded or cheaper path is
            indistinguishable from the reference one.
        """
        from vocal_helper.asr_cloud import GLADIA_PRERECORDED_MODEL

        asr = (
            f"gladia:{GLADIA_PRERECORDED_MODEL}"
            if self.config.transcriber == "gladia"
            else "elevenlabs:scribe"
        )
        return f"{asr}+{self.config.diarizer}" if self.config.diarizer else asr

    def run(self, audio: str) -> list[CloudUtterance]:
        """Transcribe and diarize ``audio``.

        Parameters
        ----------
        audio : str
            Media path. Any container; the waveform is what gets uploaded.

        Returns
        -------
        list of CloudUtterance
            In time order, each naming the engine that produced it.

        Raises
        ------
        RuntimeError
            When a required credential is missing, or a provider refuses the call.
            Never a silent fallback to a local engine.
        """
        import os_helper as osh

        config = self.config
        engine = self.engine_name()

        if config.transcriber == "gladia":
            osh.info(f"  cloud: transcribing with {engine.split('+')[0]}")
            utterances = _from_gladia(
                gladia_utterances(
                    audio,
                    languages=config.languages,
                    code_switching=config.code_switching,
                    custom_vocabulary=config.glossary,
                    overlap_s=config.overlap_s,
                ),
                engine,
            )
        else:
            osh.info("  cloud: transcribing with elevenlabs:scribe")
            payload = elevenlabs_transcribe(
                audio=audio,
                language=(config.languages or [None])[0],
                keyterms=config.glossary,
                diarize=not config.diarizer,  # no need when pyannoteAI will relabel
                num_speakers=config.num_speakers,
            )
            utterances = _from_elevenlabs(payload, engine)

        if config.diarizer == "pyannoteai":
            osh.info("  cloud: diarizing with pyannoteai")
            segments = pyannoteai_diarize(
                audio=audio,
                num_speakers=config.num_speakers,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers,
            )
            utterances = relabel_utterances(utterances, segments, engine=engine)

        osh.info(f"  cloud: {len(utterances)} utterance(s) from {engine}")
        return utterances


class CloudLivePipeline:
    """Paid live transcription and diarization: one PCM stream, two providers, fused.

    The topology is a **fan-out**, not a chain: the same audio goes to Gladia (for
    words) and to pyannoteAI Live-1 (for speakers), concurrently with receiving from
    both, and the two answers are joined at the end. It has to be this way, because
    Gladia does not diarize live.

    Both connections are **injected** rather than opened here, which is what makes
    this class testable against a local mock server and keeps credentials out of it.
    A caller owns the ``async with websockets.connect(...)`` scope for each, using
    the URLs from :func:`vocal_helper.asr_cloud.open_gladia_live_session` and
    :func:`vocal_helper.diar_cloud.open_pyannoteai_live_stream`.

    Parameters
    ----------
    asr_ws : Any
        Open connection to the transcription socket.
    diar_ws : Any
        Open connection to the diarization socket.
    pre_send : callable, optional
        ``async (ndarray) -> ndarray``, applied to every frame **before** it leaves
        the machine. Identity by default. This is the anonymisation injection
        point, and the reason it lives here is a separation of duties: this library
        provides transport and a hook, and never personal-data policy, which
        belongs to the caller who knows what must be removed.

    Examples
    --------
    >>> import asyncio, json, numpy as np
    >>> class FakeWs:
    ...     def __init__(self, replies): self.replies, self.sent = replies, []
    ...     async def send(self, payload): self.sent.append(payload)
    ...     def __aiter__(self):
    ...         async def gen():
    ...             for r in self.replies: yield json.dumps(r)
    ...         return gen()
    >>> async def demo():
    ...     asr = FakeWs([{"type": "final", "words": [
    ...         {"word": "oui", "time_begin": 0.1, "time_end": 0.4}]}])
    ...     diar = FakeWs([{"type": "diarization_speaker_start",
    ...                     "data": {"timestamp": 0.0, "speaker": "S0"}},
    ...                    {"type": "diarization_speaker_end",
    ...                     "data": {"timestamp": 1.0, "speaker": "S0"}}])
    ...     async def source():
    ...         yield np.zeros(1600, dtype=np.float32)
    ...     out = await CloudLivePipeline(asr, diar).run(source())
    ...     return [(u["speaker"], u["text"]) for u in out], len(asr.sent), len(diar.sent)
    >>> asyncio.run(demo())
    ([('S0', 'oui')], 1, 1)
    """

    def __init__(self, asr_ws: Any, diar_ws: Any, *, pre_send: Any = None) -> None:
        self._asr = GladiaLiveASR(asr_ws)
        self._diar = PyannoteAiLiveDiar(diar_ws)
        self._pre_send = pre_send

    async def run(
        self,
        pcm_source: AsyncIterator[Any],
        *,
        stop: asyncio.Event | None = None,
    ) -> list[CloudUtterance]:
        """Fan ``pcm_source`` out to both providers and return the fused utterances.

        Parameters
        ----------
        pcm_source : async iterator of ndarray
            Mono float32 chunks, from any of :mod:`vocal_helper.sources`.
        stop : asyncio.Event, keyword-only, optional
            Set externally — a "stop" button, a maximum-duration timer — to end the
            session without waiting for the source to exhaust itself. Neither a
            microphone nor an open-ended live URL ends on its own, so a live session
            without this has no way to finish.

        Returns
        -------
        list of CloudUtterance
            The fused record. Partial transcripts are excluded: they get revised,
            and a record must not contain retracted words.

        Notes
        -----
        Three tasks run concurrently: sending audio, draining transcripts, draining
        speaker events. The two drain loops end when their socket closes, and this
        method does **not** close either one — the caller owns that scope. End of
        stream is signalled to the diarizer (which documents a message for it);
        Gladia documents no equivalent, so closing the socket is what ends it.
        """
        transcripts: list = []
        events: list = []

        async def send_audio() -> None:
            async for pcm in pcm_source:
                if stop is not None and stop.is_set():
                    break
                frame = await self._pre_send(pcm) if self._pre_send is not None else pcm
                await asyncio.gather(
                    self._asr.send_audio(_asr_wire(frame)),
                    self._diar.send_audio(_diar_wire(frame)),
                )

        async def drain_asr() -> None:
            async for transcript in self._asr.transcripts():
                transcripts.append(transcript)

        async def drain_diar() -> None:
            async for event in self._diar.events():
                events.append(event)

        await asyncio.gather(send_audio(), drain_asr(), drain_diar())
        return fuse_live_transcripts(transcripts, events)
