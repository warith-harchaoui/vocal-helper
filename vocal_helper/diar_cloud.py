"""
vocal_helper.diar_cloud
=======================

Hosted diarization: **pyannoteAI**, pre-recorded and live. The paid half of the
diarization stage, mirroring :mod:`vocal_helper.diar` on the other side of the
network.

Extracted from ``notes_helper.cloud_transcribe`` and
``notes_helper.live_cloud.diar_cloud`` (same author), where this code was written
against the real APIs.

Why the paid diarizer is the one worth paying for
--------------------------------------------------
Measured on this suite's own material, the free local path does not lose on the
**text** — its transcription confidence sits close to the hosted one. It collapses
on the **speakers**: a six-voice broadcast panel came back as roughly two
clusters, hundreds of turns merged into one.

Diarization is also, by a wide margin, the cheaper of the two hosted stages. So
the interesting purchase is not "buy the cloud", it is **buy the diarization and
keep the local transcription** — the hybrid that
:class:`vocal_helper.cloud.CloudOfflinePipeline` exposes as a first-class mode,
because an economy of that size has no business hiding in a configuration key.

Why no speaker count is passed by default
-----------------------------------------
Discovering how many people are talking is exactly what this provider is good at,
and pinning the count can wedge the segmentation on a wrong one. Left alone, the
call is a pure function of the audio, so the ledger keys on the bytes alone and a
recording diarizes once, ever.

Three opt-in hints exist, all part of the ledger key so a hinted run is cached
separately from the auto-detected one, and so the hint actually takes effect:

- ``min_speakers`` — **prefer this one**. A roster is a floor, never a cap: these
  voices are known to exist, and an unnamed extra one stays possible. It lets the
  provider keep separating genuinely distinct voices instead of merging them to
  fit a number.
- ``max_speakers`` — an upper bound, for when over-splitting (one voice fragmented
  across several clusters) is the known failure instead.
- ``num_speakers`` — an exact count. The most constraining of the three: it will
  merge or split to hit the number even when the audio disagrees. Use it only when
  the total is genuinely fixed and an extra voice is impossible.

Which model, and why it is named here rather than left to a default
-------------------------------------------------------------------
Verified against the provider's documentation and the installed client on
2026-11-01, because the answer had changed since this code was first written.

**Batch: ``precision-3``.** It is the provider's most accurate batch model, and
the only one with tunable voice-activity and cross-talk sensitivity. The client
library defaults to ``precision-2``, the previous generation — so a caller who says
nothing pays the full price for the older model. That is why the default is set
here explicitly: on a paid call, taking a library's default is not a neutral act.
Published comparisons give precision-3 as 34.2 % more accurate than community-1
against 28 % for precision-2, and the provider's own launch note puts precision-3
at 10.4 % lower average error than precision-2.

**Live: ``live-1``.** Not a preference — it is the only streaming model the service
offers.

One consequence for the ledger: the **model is part of the billing key**. A run
asking for precision-3 must not be served a cached precision-2 result, or the
upgrade would be invisible and unpaid-for in the worst way: the better model
appears to have changed nothing.

Note that ``confidence`` is not supported by precision-3 (``turnLevelConfidence``
is), so a caller wanting per-segment confidence has to choose between the newest
model and that one output.

Live-1's envelope, which the caller must respect
------------------------------------------------
16 kHz mono, chunks of about 100 ms, under 300 ms of latency, at most 8 speakers
and at most 10 parallel streams, for sessions up to 5 hours. It is
**diarization-only**: the official pattern is to send the same stream to a
transcriber and to Live-1, then fuse — which is what
:mod:`vocal_helper.fusion` does, and why it is not optional.

Usage example
-------------
>>> parse_pyannote_message({"type": "diarization_speaker_start",
...                         "data": {"timestamp": 1.5, "speaker": "SPEAKER_00"}})
DiarEvent(kind='start', timestamp=1.5, speaker='SPEAKER_00', message=None)
>>> parse_pyannote_message({"type": "keepalive"}) is None
True
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from vocal_helper.cloud_media import upload_ready
from vocal_helper.credentials import credential
from vocal_helper.wallet import paid

__all__ = [
    "BATCH_MODEL",
    "LIVE_MODEL",
    "MAX_PARALLEL_STREAMS",
    "MAX_SESSION_HOURS",
    "MAX_SPEAKERS",
    "PCM_CHUNK_BYTES",
    "PCM_CHUNK_MS",
    "PCM_SAMPLE_RATE",
    "DiarEvent",
    "PyannoteAiLiveDiar",
    "open_pyannoteai_live_stream",
    "parse_pyannote_message",
    "pcm_f32_to_wire_bytes",
    "pyannoteai_diarize",
]

BATCH_MODEL = "precision-3"
"""The most accurate batch model. The client library would default to ``precision-2``."""

LIVE_MODEL = "live-1"
"""The only streaming model the service offers. A constraint, not a choice."""

PCM_SAMPLE_RATE = 16_000
"""Hz — the only sample rate Live-1 accepts."""

PCM_CHUNK_MS = 100
"""The cadence the protocol is tuned for (not enforced client-side)."""

PCM_CHUNK_BYTES = 6_400
"""1600 samples × 4 bytes (float32) mono at 16 kHz, i.e. one 100 ms chunk."""

MAX_SPEAKERS = 8
MAX_PARALLEL_STREAMS = 10
MAX_SESSION_HOURS = 5.0


# --------------------------------------------------------------------------- #
#  Live (WebSocket): pyannoteAI Live-1
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiarEvent:
    """One speaker-turn boundary, parsed and transport-agnostic."""

    kind: str
    """``"start"``, ``"end"`` or ``"error"``."""

    timestamp: float | None
    speaker: str | None
    message: str | None = None


def parse_pyannote_message(raw: dict[str, Any]) -> DiarEvent | None:
    """Parse one decoded Live-1 message, or ``None`` when it is not an event.

    Pure: no input/output, so the wire format is testable without a socket.

    Parameters
    ----------
    raw : dict
        One decoded JSON message.

    Returns
    -------
    DiarEvent or None
        ``None`` for messages that carry no boundary and no error.

    Examples
    --------
    >>> parse_pyannote_message({"type": "diarization_speaker_end",
    ...                         "data": {"timestamp": 9.0, "speaker": "S1"}}).kind
    'end'
    >>> parse_pyannote_message({"type": "error", "message": "quota"}).message
    'quota'
    """
    kind = raw.get("type")
    if kind in ("diarization_speaker_start", "diarization_speaker_end"):
        data = raw.get("data") or {}
        return DiarEvent(
            "start" if kind == "diarization_speaker_start" else "end",
            data.get("timestamp"),
            data.get("speaker"),
        )
    if kind == "error":
        return DiarEvent("error", None, None, message=raw.get("message"))
    return None


def open_pyannoteai_live_stream() -> dict[str, str]:
    """Open a Live-1 session and return its identifier and socket URL.

    Returns
    -------
    dict of str
        ``{"id", "url"}``.

    Raises
    ------
    RuntimeError
        When ``PYANNOTEAI_API_KEY`` is not configured.
    """
    import requests

    key = credential("PYANNOTEAI_API_KEY")
    if not key:
        raise RuntimeError(
            "PYANNOTEAI_API_KEY missing: set the environment variable, or the "
            "credentials: section of settings.yaml / credentials.yaml. The paid live "
            "mode stays off until it is configured."
        )
    resp = requests.post(
        "https://api.pyannote.ai/v1/live",
        headers={"Authorization": f"Bearer {key}"},
        # The endpoint exposes exactly one streaming model, so there is nothing to
        # select; naming it keeps the request self-documenting if that ever changes.
        json={"model": LIVE_MODEL},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


class PyannoteAiLiveDiar:
    """Stream PCM over an open WebSocket and yield parsed speaker events.

    The connection is **injected**, never opened here, for the same reason as
    :class:`vocal_helper.asr_cloud.GladiaLiveASR`: production passes a real
    connection, a test passes a mock, and this class touches neither the network
    nor a credential.

    Parameters
    ----------
    ws : Any
        An open connection exposing ``await ws.send(...)`` and async iteration.

    Examples
    --------
    >>> import asyncio, json
    >>> class FakeWs:
    ...     def __init__(self): self.sent = []
    ...     async def send(self, payload): self.sent.append(payload)
    ...     def __aiter__(self):
    ...         async def gen():
    ...             yield json.dumps({"type": "diarization_speaker_start",
    ...                               "data": {"timestamp": 0.0, "speaker": "S0"}})
    ...         return gen()
    >>> async def demo():
    ...     ws = FakeWs()
    ...     diar = PyannoteAiLiveDiar(ws)
    ...     await diar.end_stream()
    ...     return [e.speaker async for e in diar.events()], ws.sent
    >>> asyncio.run(demo())
    (['S0'], ['{"type": "end_of_stream"}'])
    """

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def send_audio(self, pcm_f32le: bytes) -> None:
        """Send one raw PCM chunk; Live-1 expects roughly 100 ms per chunk."""
        await self._ws.send(pcm_f32le)

    async def end_stream(self) -> None:
        """Signal end of stream, so the server flushes its final events and closes."""
        await self._ws.send(json.dumps({"type": "end_of_stream"}))

    async def events(self) -> AsyncIterator[DiarEvent]:
        """Yield each speaker event as the server emits it, until the socket closes."""
        async for raw in self._ws:
            if isinstance(raw, (bytes, bytearray)):
                continue  # a binary frame from the server would be unexpected
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                continue
            event = parse_pyannote_message(message)
            if event is not None:
                yield event


# --------------------------------------------------------------------------- #
#  Pre-recorded: pyannoteAI batch
# --------------------------------------------------------------------------- #


@paid(
    "pyannoteai.diarize",
    key_fields=("model", "num_speakers", "min_speakers", "max_speakers"),
)
def pyannoteai_diarize(
    *,
    audio: str,
    model: str = BATCH_MODEL,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    vad_sensitivity: float | None = None,
    crosstalk_sensitivity: float | None = None,
) -> list[dict]:
    """Diarize one file with pyannoteAI. Billed once per audio, model and hint set.

    Parameters
    ----------
    audio : str, keyword-only
        Media path. Converted to 16 kHz mono MP3 before upload, so a video
        recording does not send its video track for no benefit.
    model : str, optional
        Defaults to :data:`BATCH_MODEL` (``"precision-3"``), the most accurate one.
        The client library's own default is the previous generation, which is why
        this is set here rather than omitted. Part of the ledger key, so an
        upgrade is not served a stale cheaper result.
    num_speakers, min_speakers, max_speakers : int, optional
        The three opt-in hints. See the module docstring for which to reach for;
        the short version is **prefer ``min_speakers``**.
    vad_sensitivity, crosstalk_sensitivity : float, optional
        ``precision-3`` only: how eagerly speech and overlapping speech are
        detected. Left unset unless asked for, because the provider's defaults are
        tuned and a guess here moves every segment boundary at once.

    Returns
    -------
    list of dict
        Segments, each with ``start``, ``end`` and ``speaker``.

    Raises
    ------
    RuntimeError
        When the key is missing, or the response carries no segments. The result
        document's shape has moved between versions, so three known layouts are
        tried before giving up — and giving up says so rather than returning an
        empty list, which would read downstream as "nobody spoke".
    """
    import os_helper as osh
    from pyannoteai.sdk import Client

    key = credential("PYANNOTEAI_API_KEY")
    if not key:
        raise RuntimeError(
            "PYANNOTEAI_API_KEY missing: set the environment variable, or the "
            "credentials: section of settings.yaml / credentials.yaml."
        )
    client = Client(token=key)

    with osh.temporary_folder(prefix="voh-upload") as tmp_dir:
        media = client.upload(upload_ready(audio, tmp_dir))

    hints: dict[str, Any] = {"model": model}
    if num_speakers:
        hints["num_speakers"] = num_speakers
    if min_speakers:
        hints["min_speakers"] = min_speakers
    if max_speakers:
        hints["max_speakers"] = max_speakers
    # precision-3's own knobs travel as extra body arguments, which the client
    # forwards verbatim; sending them to an older model would be rejected, so they
    # are only ever included when a caller asked for them.
    if vad_sensitivity is not None:
        hints["vadSensitivity"] = vad_sensitivity
    if crosstalk_sensitivity is not None:
        hints["crosstalkSensitivity"] = crosstalk_sensitivity

    result = client.retrieve(client.diarize(media, **hints))
    for path in (("output", "diarization"), ("diarization",), ("output", "segments")):
        cursor: Any = result
        try:
            for step in path:
                cursor = cursor[step]
            return cursor
        except (KeyError, TypeError):
            continue
    raise RuntimeError("pyannoteAI returned no diarization segments")


def pcm_f32_to_wire_bytes(pcm: Any) -> bytes:
    """Mono float32 samples to Live-1's ``f32le`` wire format.

    A reinterpretation, not a conversion: the source frame contract already yields
    float32, so there is no resampling and no requantisation to get wrong here.

    Parameters
    ----------
    pcm : numpy.ndarray
        Mono float32 samples.

    Returns
    -------
    bytes
        Little-endian 32-bit floats.

    Examples
    --------
    >>> import numpy as np
    >>> len(pcm_f32_to_wire_bytes(np.zeros(1600, dtype=np.float32))) == PCM_CHUNK_BYTES
    True
    """
    import numpy as np

    return np.asarray(pcm, dtype=np.float32).tobytes()
