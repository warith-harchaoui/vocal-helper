"""
Tests for the NeMo Sortformer output parser (``_parse_sortformer_segments``).

Regression guard for the bug where the offline NeMo backend returned **no
speakers at all**: nemo-toolkit 2.x emits the compact ``"<start> <end>
<speaker>"`` form, but the old parser only understood legacy RTTM and dropped
every line — silently producing an empty diarization (DER 1.0). Model-free.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

from vocal_helper.diar import _parse_sortformer_segments


def test_parses_compact_start_end_speaker() -> None:
    """The nemo 2.x compact form is parsed into (t0, t1, speaker)."""
    # Three cues, two speakers — the exact shape nemo-toolkit 2.x emits; the
    # parser must read each as absolute (start, end), not (start, duration).
    lines = ["0.000 1.760 speaker_0", "1.920 3.040 speaker_1", "4.400 8.080 speaker_0"]
    assert _parse_sortformer_segments(lines) == [
        (0.0, 1.76, "speaker_0"),
        (1.92, 3.04, "speaker_1"),
        (4.4, 8.08, "speaker_0"),
    ]


def test_parses_legacy_rttm() -> None:
    """Legacy RTTM ``SPEAKER`` lines still parse (start + duration)."""
    # RTTM field 4 is start and field 5 is *duration*, so 0.5 + 2.0 → end 2.5.
    # Keeping this path means an older nemo build never silently drops speakers.
    lines = ["SPEAKER meeting 1 0.500 2.000 <NA> <NA> spk1 <NA> <NA>"]
    assert _parse_sortformer_segments(lines) == [(0.5, 2.5, "spk1")]


def test_skips_malformed_and_nonstring() -> None:
    """Garbage, empty, and non-string entries are dropped, not crashed on."""
    # Mixed junk exercises every guard at once: non-float tokens, non-string
    # entries, and a zero/negative-span cue must all be filtered, never raised.
    lines = ["", "not a segment line", "a b c", None, 42, "1.0 0.5 spk"]
    # "a b c" -> non-float; None/42 -> non-string; "1.0 0.5 spk" -> t1<=t0 dropped.
    assert _parse_sortformer_segments(lines) == []


def test_empty_input_yields_no_segments() -> None:
    """An empty backend output yields an empty diarization, not an error."""
    # The degenerate case the original bug turned into DER 1.0 — pin it explicitly.
    assert _parse_sortformer_segments([]) == []
