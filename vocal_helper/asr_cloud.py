"""
vocal_helper.asr_cloud
======================

Hosted transcription: **Gladia** (pre-recorded and live) and **ElevenLabs Scribe**
(pre-recorded). The paid half of the transcription stage, mirroring
:mod:`vocal_helper.asr` on the other side of the network.

Extracted from ``notes_helper.cloud_transcribe`` and
``notes_helper.live_cloud.asr_cloud`` (same author), where this code was written,
measured and debugged against the real APIs. It belongs in the library that owns
transcription rather than in a product sitting on top of one.

Opt-in, never a fallback
------------------------
A missing key raises. There is no path where an absent credential degrades to a
local engine, and no path where a present credential turns a free run into a
billed one: the caller asks for a cloud mode by name or gets the local one. Every
billed call goes through :func:`vocal_helper.wallet.paid`, so the same audio is
never charged twice.

Two models, two endpoints, and the mistake that is easy to make
---------------------------------------------------------------
Gladia exposes transcription twice, and the model names are **not**
interchangeable.

- **Pre-recorded** (``POST /v2/upload`` then ``/v2/pre-recorded``) accepts
  ``solaria-3``, the newer model, optimised for English, French, German, Spanish
  and Italian, and it **does** diarize.
- **Live** (``POST /v2/live`` then a WebSocket) accepts ``solaria-1`` and nothing
  else: the specification's streaming-model enumeration has a single member. An
  earlier version of this code set the live model to ``solaria-3`` on the
  reasonable-sounding assumption that newer is better; the API rejects it.

And the premise of the whole live design: **Gladia does not diarize live**
(confirmed against Gladia's own documentation — real-time diarization is not
supported). Speaker identity for the live paid mode must come from pyannoteAI
Live-1 in :mod:`vocal_helper.diar_cloud` and be fused in
:mod:`vocal_helper.fusion`, never from Gladia alone.

Neither live endpoint has a Python software development kit; both are
wire-protocol only, which is why the clients below are hand-written.

Why a glossary is not part of the ledger key
--------------------------------------------
Both providers accept a list of domain terms to bias recognition toward: Gladia
calls it a custom vocabulary, ElevenLabs calls it keyterm prompting. Neither joins
the ledger key. Editing a glossary and re-running the **same audio** reuses the
prior transcript rather than paying again: a misspelt proper noun is cheaper to
repair downstream than to re-transcribe, and when a quota is spent, a cache miss
does not merely cost money — it silently fails over and changes the result.

Bias strength matters too. At Gladia's default intensity the model force-fits
boosted terms, "hearing" a glossary proper noun where none was spoken. The default
here is lower, which keeps the spelling help and cuts the hallucinated insertions.

Usage example
-------------
>>> parse_gladia_message({"type": "final", "utterances": [
...     {"words": [{"word": "bonjour", "time_begin": 0.0,
...                 "time_end": 0.6, "confidence": 0.98}]}]})
AsrTranscript(kind='final', words=[AsrWord(word='bonjour', time_begin=0.0, time_end=0.6, confidence=0.98)])
>>> parse_gladia_message({"type": "session_start"}) is None
True
>>> sanitize_keyterms(["Arcom", "a (very) long & messy term", "x"])
['Arcom', 'a long et messy term']
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from vocal_helper.cloud_media import (
    extract_clip,
    match_speakers_across_overlap,
    plan_chunks,
    upload_ready,
)
from vocal_helper.credentials import credential
from vocal_helper.wallet import paid

__all__ = [
    "ELEVENLABS_STT_URL",
    "GLADIA_LIVE_MODEL",
    "GLADIA_MAX_S",
    "GLADIA_PRERECORDED_MODEL",
    "PCM_SAMPLE_RATE",
    "AsrTranscript",
    "AsrWord",
    "GladiaLiveASR",
    "elevenlabs_transcribe",
    "gladia_transcribe",
    "gladia_utterances",
    "open_gladia_live_session",
    "parse_gladia_message",
    "pcm_f32_to_wire_bytes",
    "sanitize_keyterms",
]

PCM_SAMPLE_RATE = 16_000
"""Hz. Gladia also accepts 8000/32000/44100/48000, declared at session init."""

GLADIA_PRERECORDED_MODEL = "solaria-3"
"""Pre-recorded default. Override with ``VOH_GLADIA_MODEL`` (``solaria-1`` for 100+ languages)."""

GLADIA_LIVE_MODEL = "solaria-1"
"""The ONLY model the live endpoint accepts. Not a preference — a constraint."""

GLADIA_MAX_S = 8000.0
"""Just under Gladia's ~8100 s (~135 min) per-file cap; longer media is chunked and stitched."""

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


# --------------------------------------------------------------------------- #
#  Live (WebSocket): Gladia solaria-1
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AsrWord:
    """One transcribed word with its span and the model's confidence in it."""

    word: str
    time_begin: float
    time_end: float
    confidence: float = 1.0


@dataclass(frozen=True)
class AsrTranscript:
    """One parsed live transcript event, partial or final."""

    kind: str
    """``"partial"`` (for display) or ``"final"`` (for the record)."""

    words: list[AsrWord] = field(default_factory=list)


def parse_gladia_message(raw: dict[str, Any]) -> AsrTranscript | None:
    """Parse one decoded live message, or ``None`` when it is not a transcript.

    Pure: no input/output, so the wire format is testable without a socket.

    Parameters
    ----------
    raw : dict
        One decoded JSON message from the live WebSocket.

    Returns
    -------
    AsrTranscript or None
        ``None`` for anything that is not a transcript event (session lifecycle,
        acknowledgements), so a caller can ignore the rest without a type check.

    Examples
    --------
    >>> parse_gladia_message({"type": "partial", "words": [
    ...     {"word": "oui", "time_begin": 1.0, "time_end": 1.2}]}).kind
    'partial'
    >>> parse_gladia_message({"type": "final", "utterances": []}).words
    []
    """
    kind = raw.get("type")
    if kind not in ("partial", "final"):
        return None
    words: list[AsrWord] = []
    # Some payloads nest words under ``utterances``, others flatten them onto the
    # message; fall back to the message itself so both shapes parse.
    for utterance in raw.get("utterances") or [raw]:
        for word in utterance.get("words") or []:
            try:
                words.append(
                    AsrWord(
                        word=word["word"],
                        time_begin=float(word["time_begin"]),
                        time_end=float(word["time_end"]),
                        confidence=float(word.get("confidence", 1.0)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                # One malformed word must not discard the rest of the utterance.
                continue
    return AsrTranscript(kind, words)


def open_gladia_live_session(*, sample_rate: int = PCM_SAMPLE_RATE) -> dict[str, str]:
    """Open a Gladia live session and return its identifiers and socket URL.

    Parameters
    ----------
    sample_rate : int, keyword-only, optional
        The rate the stream will actually carry. Declared here, honoured by the
        socket; mismatching it corrupts the audio silently.

    Returns
    -------
    dict of str
        ``{"id", "created_at", "url"}``. The ``url`` carries a temporary token.

    Raises
    ------
    RuntimeError
        When ``GLADIA_API_KEY`` is not configured. The mode is unavailable rather
        than degraded.
    """
    import requests

    key = credential("GLADIA_API_KEY")
    if not key:
        raise RuntimeError(
            "GLADIA_API_KEY missing: set the environment variable, or the credentials: "
            "section of settings.yaml / credentials.yaml. The paid live mode stays off "
            "until it is configured."
        )
    resp = requests.post(
        "https://api.gladia.io/v2/live",
        headers={"x-gladia-key": key},
        json={
            "encoding": "wav/pcm",
            "bit_depth": 16,
            "sample_rate": sample_rate,
            "channels": 1,
            "model": GLADIA_LIVE_MODEL,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


class GladiaLiveASR:
    """Stream PCM over an open WebSocket and yield parsed transcripts.

    The connection is **injected**, never opened here: production passes a real
    ``websockets`` connection to the URL from :func:`open_gladia_live_session`, and
    a test points the same class at a local mock server. So this class touches
    neither the network, nor a credential, nor the optional dependency — it only
    uses the ``send`` / async-iteration interface any such connection provides.

    Parameters
    ----------
    ws : Any
        An open connection exposing ``await ws.send(bytes)`` and async iteration
        over received frames.

    Examples
    --------
    >>> import asyncio
    >>> class FakeWs:
    ...     def __init__(self): self.sent = []
    ...     async def send(self, payload): self.sent.append(payload)
    ...     def __aiter__(self):
    ...         async def gen():
    ...             yield json.dumps({"type": "final", "words": [
    ...                 {"word": "salut", "time_begin": 0.0, "time_end": 0.4}]})
    ...         return gen()
    >>> async def demo():
    ...     ws = FakeWs()
    ...     asr = GladiaLiveASR(ws)
    ...     await asr.send_audio(b"\\x00\\x00")
    ...     return [t.words[0].word async for t in asr.transcripts()], ws.sent
    >>> asyncio.run(demo())
    (['salut'], [b'\\x00\\x00'])
    """

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def send_audio(self, pcm: bytes) -> None:
        """Send one raw PCM chunk, in the encoding declared at session creation."""
        await self._ws.send(pcm)

    async def transcripts(self) -> AsyncIterator[AsrTranscript]:
        """Yield each transcript event as the server emits it, until the socket closes."""
        async for raw in self._ws:
            if isinstance(raw, (bytes, bytearray)):
                continue  # a binary frame from the server would be unexpected
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                continue
            event = parse_gladia_message(message)
            if event is not None:
                yield event


# --------------------------------------------------------------------------- #
#  Pre-recorded: Gladia solaria-3
# --------------------------------------------------------------------------- #


@paid("gladia.transcribe")
def gladia_transcribe(
    *,
    audio: str,
    languages: list[str] | None = None,
    code_switching: bool = True,
    custom_vocabulary: list[str] | None = None,
) -> dict:
    """Transcribe one file with Gladia's pre-recorded endpoint. Billed once per audio.

    Parameters
    ----------
    audio : str, keyword-only
        Media path. Converted to 16 kHz mono MP3 before upload.
    languages : list of str, optional
        Expected languages. ``None`` lets Gladia detect.
    code_switching : bool, optional
        Whether a single recording may switch between the declared languages.
    custom_vocabulary : list of str, optional
        Domain terms and proper nouns to bias toward. **Not** part of the ledger
        key — see the module docstring.

    Returns
    -------
    dict
        Gladia's full result document.

    Raises
    ------
    RuntimeError
        When the key is missing, or when the upload, the job submission or the job
        itself fails. Each case says which, with the provider's own message: a
        rejected oversized vocabulary and an exhausted quota are different
        problems and must not read alike.
    """
    import mimetypes

    import os_helper as osh
    import requests

    key = credential("GLADIA_API_KEY")
    if not key:
        raise RuntimeError(
            "GLADIA_API_KEY missing: set the environment variable, or the credentials: "
            "section of settings.yaml / credentials.yaml."
        )
    headers = {"x-gladia-key": key}

    with osh.temporary_folder(prefix="voh-upload") as tmp_dir:
        path = upload_ready(audio, tmp_dir)
        _, base, ext = osh.folder_name_ext(path)
        name = f"{base}.{ext}" if ext else base
        megabytes = osh.size_file(path) // (1 << 20)
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            resp = requests.post(
                "https://api.gladia.io/v2/upload",
                headers=headers,
                files={"audio": (name, fh, mime)},
                timeout=1800,
            )

    uploaded = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if "audio_url" not in uploaded:
        raise RuntimeError(
            f"Gladia upload failed ({resp.status_code}) for {name} [{megabytes} MB]: "
            f"{str(uploaded) or resp.text[:300]}"
        )

    config: dict[str, Any] = {
        "audio_url": uploaded["audio_url"],
        "diarization": True,
        "model": os.environ.get("VOH_GLADIA_MODEL", GLADIA_PRERECORDED_MODEL),
    }
    if languages:
        config["language_config"] = {"languages": list(languages), "code_switching": code_switching}
    if custom_vocabulary:
        # An oversized vocabulary is rejected wholesale, with no job identifier, so cap
        # it here and say so rather than lose the entire request to one long glossary.
        cap = int(os.environ.get("VOH_GLADIA_MAX_VOCAB", "1000"))
        vocabulary = list(custom_vocabulary)[:cap]
        if len(custom_vocabulary) > cap:
            osh.warning(
                f"  gladia: custom vocabulary {len(custom_vocabulary)} > cap {cap}; "
                f"sending the first {cap}"
            )
        config["custom_vocabulary"] = vocabulary
        config["custom_vocabulary_config"] = {
            "default_intensity": float(os.environ.get("VOH_GLADIA_VOCAB_INTENSITY", "0.3")),
        }

    submit = requests.post(
        "https://api.gladia.io/v2/pre-recorded",
        headers={**headers, "Content-Type": "application/json"},
        json=config,
        timeout=60,
    )
    job = submit.json() if submit.headers.get("content-type", "").startswith("application/json") else {}
    if "id" not in job:
        raise RuntimeError(
            f"Gladia rejected the job ({submit.status_code}): {str(job) or submit.text[:300]}"
        )

    import time

    while True:
        result = requests.get(
            f"https://api.gladia.io/v2/pre-recorded/{job['id']}", headers=headers, timeout=60
        ).json()
        status = result.get("status")
        if status == "error":
            raise RuntimeError(f"Gladia job failed: {result}")
        if status == "done":
            return result
        time.sleep(10)


def gladia_utterances(
    audio: str,
    *,
    languages: list[str] | None = None,
    code_switching: bool = True,
    custom_vocabulary: list[str] | None = None,
    overlap_s: float = 120.0,
) -> list[dict]:
    """Gladia utterances for a recording of **any** length.

    A file under the cap is one call. A longer one is split into overlapping
    chunks, each transcribed separately, then stitched: chunk times are shifted to
    absolute, each chunk's speaker labels are matched to the running result through
    the shared window, and only the turns past the overlap's midpoint are appended,
    so the seam neither loses nor duplicates anything.

    Parameters
    ----------
    audio : str
        Media path.
    languages, code_switching, custom_vocabulary
        Passed through to :func:`gladia_transcribe`.
    overlap_s : float, keyword-only, optional
        How much consecutive chunks share. Long enough for several speaker turns,
        or the label matching has nothing to measure.

    Returns
    -------
    list of dict
        Gladia utterances in absolute time, with globally consistent speakers.
    """
    import os_helper as osh

    from vocal_helper.cloud_media import audio_duration

    duration = audio_duration(audio)
    offsets = plan_chunks(duration_s=duration, cap_s=GLADIA_MAX_S, overlap_s=overlap_s)

    if len(offsets) == 1:
        result = gladia_transcribe(
            audio=audio,
            languages=languages,
            code_switching=code_switching,
            custom_vocabulary=custom_vocabulary,
        )
        return result["result"]["transcription"]["utterances"]

    osh.info(
        f"  gladia: {duration / 60:.0f} min over the {GLADIA_MAX_S / 60:.0f} min cap "
        f"→ {len(offsets)} overlapping chunks, stitched on {overlap_s:.0f} s of shared audio"
    )
    merged: list[dict] = []
    for index, offset in enumerate(offsets):
        length = min(GLADIA_MAX_S, duration - offset)
        with osh.temporary_folder(prefix="voh-chunk") as tmp_dir:
            clip = extract_clip(audio, offset, length, osh.join(tmp_dir, "chunk.mp3"))
            result = gladia_transcribe(
                audio=clip,
                languages=languages,
                code_switching=code_switching,
                custom_vocabulary=custom_vocabulary,
            )
        utterances = result["result"]["transcription"]["utterances"]
        for utterance in utterances:  # chunk-relative → absolute
            utterance["start"] = float(utterance["start"]) + offset
            utterance["end"] = float(utterance["end"]) + offset
        if index == 0:
            merged = utterances
            continue
        mapping = match_speakers_across_overlap(merged, utterances, offset, offset + overlap_s)
        for utterance in utterances:
            # An unmatched label gets a chunk-scoped name rather than a guess: a voice
            # that only speaks after the overlap genuinely has no counterpart to match.
            utterance["speaker"] = mapping.get(
                utterance["speaker"], f"c{index}s{utterance['speaker']}"
            )
        midpoint = offset + overlap_s / 2
        merged += [u for u in utterances if u["start"] >= midpoint]
    return merged


# --------------------------------------------------------------------------- #
#  Pre-recorded: ElevenLabs Scribe
# --------------------------------------------------------------------------- #

_KEYTERM_BAD_CHARS = re.compile(r"[^0-9A-Za-zÀ-ÿ '\-]")


def sanitize_keyterms(
    terms: list[str] | None,
    *,
    cap: int = 1000,
    max_len: int = 50,
    max_spaces: int = 4,
) -> list[str]:
    """Make a keyterm list safe, so one bad term never rejects the whole call.

    Scribe validates keyterms strictly and rejects the **entire** request when a
    single term breaks a rule — which, in a product with a local fallback, shows up
    as a mysteriously free run with a worse transcript. A glossary is data nobody
    controls (hand-written, model-extracted), so it is cleaned at this boundary
    rather than trusted: curly quotes normalised, parentheticals dropped, ``&``
    spelled out, unsupported characters stripped, spaces collapsed, length
    truncated, and anything still too long or too wordy discarded.

    Parameters
    ----------
    terms : list of str or None
        The raw glossary.
    cap : int, keyword-only, optional
        Maximum number of terms to keep.
    max_len : int, keyword-only, optional
        Maximum characters per term.
    max_spaces : int, keyword-only, optional
        Maximum spaces per term, i.e. ``max_spaces + 1`` words.

    Returns
    -------
    list of str
        Cleaned, de-duplicated case-insensitively, original order preserved.

    Examples
    --------
    >>> sanitize_keyterms(None)
    []
    >>> sanitize_keyterms(["Arcom", "ARCOM", "a"])
    ['Arcom']
    >>> sanitize_keyterms(["L’Arcom (le régulateur)"])
    ["L'Arcom"]
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in terms or []:
        term = str(raw).strip().replace("’", "'").replace("‘", "'")
        term = re.sub(r"\([^)]*\)", " ", term).replace("&", " et ")
        term = _KEYTERM_BAD_CHARS.sub(" ", term)
        term = re.sub(r"\s+", " ", term).strip(" -'")[:max_len].strip(" -'")
        if len(term) < 2 or term.count(" ") > max_spaces:
            continue
        folded = term.lower()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(term)
        if len(out) >= cap:
            break
    return out


@paid("elevenlabs.transcribe", key_fields=("diarize", "num_speakers"))
def elevenlabs_transcribe(
    *,
    audio: str,
    language: str | None = None,
    keyterms: list[str] | None = None,
    diarize: bool = True,
    num_speakers: int | None = None,
    model: str | None = None,
) -> dict:
    """Transcribe one file with ElevenLabs Scribe, word by word. Billed once per audio.

    Scribe **bundles** diarization into this same response, unlike Gladia, which is
    transcription-only and always pairs with a separate diarization call. So
    ``diarize`` and ``num_speakers`` genuinely change what comes back, and both
    **are** part of the ledger key: a run with a forced speaker count must not
    silently reuse an auto-detected result.

    Scribe handles long files in one call, so no chunking is needed here.

    Parameters
    ----------
    audio : str, keyword-only
        Media path. Converted to 16 kHz mono MP3 before upload.
    language : str, optional
        ISO-639 code. ``None`` auto-detects.
    keyterms : list of str, optional
        Domain terms to bias toward. Sanitised by :func:`sanitize_keyterms`;
        **not** part of the ledger key.
    diarize : bool, optional
        Ask Scribe for a per-word speaker. A caller may instead relabel with
        pyannoteAI downstream.
    num_speakers : int, optional
        Force an exact count. Part of the ledger key.
    model : str, optional
        Defaults to ``ELEVENLABS_STT_MODEL`` or ``scribe_v1``.

    Returns
    -------
    dict
        Scribe's response: a flat word stream with timings and speakers.

    Raises
    ------
    RuntimeError
        When the key is missing, or the request is refused.
    """
    import mimetypes

    import os_helper as osh
    import requests

    key = credential("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY missing: set the environment variable, or the "
            "credentials: section of settings.yaml / credentials.yaml."
        )
    model = model or credential("ELEVENLABS_STT_MODEL", "scribe_v1")

    # Multipart form: booleans as lower-case strings, keyterms as repeated fields —
    # the standard multipart encoding for an array.
    fields: list[tuple[str, str]] = [
        ("model_id", model),
        ("diarize", "true" if diarize else "false"),
        ("timestamps_granularity", "word"),
        ("tag_audio_events", "false"),  # keep (laughter)/(footsteps) out of the text
    ]
    if language:
        fields.append(("language_code", language))
    if num_speakers:
        fields.append(("num_speakers", str(num_speakers)))
    safe = sanitize_keyterms(keyterms)
    fields.extend(("keyterms", term) for term in safe)
    if keyterms and len(keyterms) != len(safe):
        osh.info(
            f"  scribe: {len(keyterms)} keyterms → {len(safe)} sent "
            f"(≤5 words, ≤50 chars, safe charset, de-duplicated, cap 1000)"
        )

    with osh.temporary_folder(prefix="voh-upload") as tmp_dir:
        path = upload_ready(audio, tmp_dir)
        _, base, ext = osh.folder_name_ext(path)
        name = f"{base}.{ext}" if ext else base
        megabytes = osh.size_file(path) // (1 << 20)
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            resp = requests.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": key},
                data=fields,
                files={"file": (name, fh, mime)},
                timeout=3600,
            )
    if resp.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs Scribe failed ({resp.status_code}) for {name} "
            f"[{megabytes} MB]: {resp.text[:300]}"
        )
    return resp.json()

def pcm_f32_to_wire_bytes(pcm: Any) -> bytes:
    """Mono float32 samples in ``[-1, 1]`` to Gladia live's 16-bit PCM wire format.

    Clamps **before** scaling, so a source that slightly overshoots ``[-1, 1]`` — a hot
    input, a resampling overshoot — cannot wrap around into a full-amplitude 16-bit
    value, which is heard as a click and transcribed as nothing.

    Parameters
    ----------
    pcm : numpy.ndarray
        Mono float32 samples.

    Returns
    -------
    bytes
        Little-endian signed 16-bit samples.

    Examples
    --------
    >>> import numpy as np
    >>> pcm_f32_to_wire_bytes(np.array([0.0, 1.0, -2.0], dtype=np.float32))
    b'\\x00\\x00\\xff\\x7f\\x01\\x80'
    """
    import numpy as np

    clamped = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return (clamped * 32767.0).astype(np.int16).tobytes()
