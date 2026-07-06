"""
vocal_helper.tts
================

Local neural text-to-speech via `Piper <https://github.com/rhasspy/piper>`_.

Piper is the open-source TTS pick of choice for offline / industrial
deployments in 2026 : runs on CPU at ~ 0.1× real-time for "medium"
quality voices, ships ONNX weights from
`rhasspy/piper-voices <https://huggingface.co/rhasspy/piper-voices>`_,
no external API call, no cloud dependency.

Why a helper, not a pipeline stage
----------------------------------
vocal-helper is transcription-first ; the natural input to a TTS is
"speak this string", not one of the typed events the pipeline already
moves. Rather than invent a synthetic queue protocol, we expose
:class:`PiperTTS` as a synthesise-on-demand helper that callers wire
into their own subscriber. Two common patterns :

1. **Echo the LLM rolling summary out loud** — subscribe to
   :class:`SummarySnapshot`, call :meth:`PiperTTS.synth_to_file` on
   each new ``summary`` body, play the resulting WAV.
2. **Repeat each utterance back** (testing / classroom captioning) —
   subscribe to :class:`Utterance`, synth, play.

Two industrial constraints carried over from
``feedback_no_voiceprint_no_streaming``:

- The synth call is **one-shot**, returning the full PCM buffer
  (no token / chunk streaming) ; matches the user's "no
  character-by-character LLM" stance.
- No voice cloning, no pre-enrolled voice prints — Piper voices are
  generic neural voices shipped publicly on Hugging Face.

Install
-------

.. code-block:: bash

    pip install 'vocal-helper[tts]'

The first :meth:`PiperTTS.synth` call downloads the requested voice
(~ 60 MB for ``medium`` quality) to ``~/.cache/piper-voices/``.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Curated default voices — both 22050 Hz, mono, ``medium`` quality
# (the Pareto pick between footprint and naturalness on the rhasspy
# benchmark page). Override via :class:`PiperTTS(voice=...)`.
DEFAULT_VOICE_EN = "en_US-amy-medium"
DEFAULT_VOICE_FR = "fr_FR-siwis-medium"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "piper-voices"
_VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def _voice_files(voice: str) -> tuple[str, str]:
    """Return (onnx_url, json_url) for a piper voice tag.

    Piper's HF layout : ``<lang>/<lang_country>/<voice_name>/<quality>/<voice>.onnx``
    e.g. ``en/en_US/amy/medium/en_US-amy-medium.onnx``.
    """
    parts = voice.split("-")
    if len(parts) < 3 or "_" not in parts[0]:
        raise ValueError(
            f"voice tag must look like ``<lang_COUNTRY>-<name>-<quality>`` ; "
            f"got {voice!r}"
        )
    lang_country = parts[0]            # e.g. "en_US"
    lang = lang_country.split("_")[0]  # e.g. "en"
    name = parts[1]
    quality = parts[2]
    base = f"{_VOICE_BASE_URL}/{lang}/{lang_country}/{name}/{quality}/{voice}"
    return f"{base}.onnx", f"{base}.onnx.json"


class PiperTTS:
    """Local Piper TTS synthesiser.

    Parameters
    ----------
    voice : str
        Piper voice tag (e.g. ``"en_US-amy-medium"`` for English,
        ``"fr_FR-siwis-medium"`` for French). The full catalogue is at
        https://huggingface.co/rhasspy/piper-voices.
    cache_dir : Path, optional
        Where to store downloaded ``.onnx`` weights. Default
        ``~/.cache/piper-voices``.
    length_scale : float
        Speech tempo multiplier — > 1 slower, < 1 faster. Default 1.0.
    noise_scale : float
        Variance of the prosody noise. Default 0.667 (Piper's recipe).
    noise_w : float
        Variance of the phoneme duration jitter. Default 0.8.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE_EN,
        *,
        cache_dir: Path | None = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        self.voice = voice
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self._voice: Any | None = None
        self._sample_rate: int | None = None

    # ----- lifecycle -----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._voice is not None:
            return
        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PiperTTS requires the piper-tts package. "
                "Install with `pip install vocal-helper[tts]`."
            ) from e
        onnx_path = self._ensure_voice_files()
        self._voice = PiperVoice.load(onnx_path)
        # piper exposes the sample rate on the loaded voice config.
        cfg = getattr(self._voice, "config", None)
        self._sample_rate = int(getattr(cfg, "sample_rate", 22050))

    def _ensure_voice_files(self) -> str:
        """Download the ``.onnx`` + ``.onnx.json`` if missing ; return path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_url, json_url = _voice_files(self.voice)
        onnx_path = self.cache_dir / f"{self.voice}.onnx"
        json_path = self.cache_dir / f"{self.voice}.onnx.json"
        if not onnx_path.exists():
            urllib.request.urlretrieve(onnx_url, onnx_path)
        if not json_path.exists():
            urllib.request.urlretrieve(json_url, json_path)
        return str(onnx_path)

    # ----- public API ---------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Sample rate of the synthesised PCM (typically 22050 Hz)."""
        self._ensure_loaded()
        assert self._sample_rate is not None
        return self._sample_rate

    def synth(self, text: str) -> NDArray[np.float32]:
        """Synthesise ``text`` ; return mono float32 PCM at :attr:`sample_rate`.

        One-shot — no streaming, no chunking. The full waveform comes
        back when synthesis is complete.
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        self._ensure_loaded()
        assert self._voice is not None

        buf = io.BytesIO()
        # Piper's synth API writes 16-bit signed PCM ; we collect the
        # bytes and convert to float32 mono so the output matches
        # vocal-helper's PCM convention.
        self._voice.synthesize(
            text,
            buf,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w=self.noise_w,
        )
        raw = np.frombuffer(buf.getvalue(), dtype=np.int16)
        return (raw.astype(np.float32) / 32768.0).copy()

    def synth_to_file(self, text: str, path: str | Path) -> Path:
        """Synthesise ``text`` and write a 16-bit WAV at ``path``."""
        import soundfile as sf  # type: ignore

        pcm = self.synth(text)
        path = Path(path)
        sf.write(path, pcm, self.sample_rate, subtype="PCM_16")
        return path
