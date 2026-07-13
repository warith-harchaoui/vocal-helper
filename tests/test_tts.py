"""PiperTTS — construction-time tests only (no model download)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vocal_helper.tts import (
    DEFAULT_VOICE_EN,
    DEFAULT_VOICE_FR,
    PiperTTS,
    _voice_files,
)


def test_tts_voice_url_layout_en() -> None:
    onnx, json_url = _voice_files("en_US-amy-medium")
    assert "en/en_US/amy/medium/en_US-amy-medium.onnx" in onnx
    assert json_url.endswith(".onnx.json")


def test_tts_voice_url_layout_fr() -> None:
    onnx, json_url = _voice_files("fr_FR-siwis-medium")
    assert "fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx" in onnx
    assert json_url.endswith(".onnx.json")


def test_tts_rejects_malformed_voice_tag() -> None:
    with pytest.raises(ValueError):
        _voice_files("not-a-voice")


def test_tts_defaults() -> None:
    """Constructor is cheap — no download, no ONNX runtime load."""
    t = PiperTTS()
    assert t.voice == DEFAULT_VOICE_EN
    assert t.cache_dir == Path.home() / ".cache" / "piper-voices"
    assert t.length_scale == 1.0
    assert t.noise_scale == pytest.approx(0.667)
    assert t.noise_w == pytest.approx(0.8)
    assert t._voice is None
    assert t._sample_rate is None


def test_tts_accepts_overrides(tmp_path: Path) -> None:
    t = PiperTTS(
        voice=DEFAULT_VOICE_FR,
        cache_dir=tmp_path,
        length_scale=1.2,
        noise_scale=0.5,
        noise_w=0.7,
    )
    assert t.voice == DEFAULT_VOICE_FR
    assert t.cache_dir == tmp_path
    assert t.length_scale == pytest.approx(1.2)
    assert t.noise_scale == pytest.approx(0.5)
    assert t.noise_w == pytest.approx(0.7)


def test_tts_empty_string_returns_empty_buffer() -> None:
    """Synthesising the empty string must not load the model
    (defensive, keeps the zero-call path fast)."""
    import numpy as np

    t = PiperTTS()
    out = t.synth("")
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)
    assert out.dtype == np.float32
    # Still no model load.
    assert t._voice is None
