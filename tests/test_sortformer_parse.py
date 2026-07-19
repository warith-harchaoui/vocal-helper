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


def test_parses_both_compact_and_legacy_rttm_forms() -> None:
    """Both the nemo 2.x compact form and legacy RTTM parse into (t0, t1, speaker).

    The compact ``"<start> <end> <speaker>"`` lines nemo-toolkit 2.x emits must
    be read as absolute (start, end), not (start, duration); the older RTTM
    ``SPEAKER`` form (field 4 start, field 5 *duration*, so 0.5 + 2.0 → 2.5) must
    still parse so an older nemo build never silently drops speakers.
    """
    # nemo 2.x compact cues — three cues, two speakers — absolute start/end.
    compact = ["0.000 1.760 speaker_0", "1.920 3.040 speaker_1", "4.400 8.080 speaker_0"]
    assert _parse_sortformer_segments(compact) == [
        (0.0, 1.76, "speaker_0"),
        (1.92, 3.04, "speaker_1"),
        (4.4, 8.08, "speaker_0"),
    ]
    # Legacy RTTM — start + duration → absolute end.
    rttm = ["SPEAKER meeting 1 0.500 2.000 <NA> <NA> spk1 <NA> <NA>"]
    assert _parse_sortformer_segments(rttm) == [(0.5, 2.5, "spk1")]


def test_drops_malformed_nonstring_and_empty_input() -> None:
    """Garbage / non-string / empty input yields an empty diarization, never a crash.

    Mixed junk exercises every guard at once — non-float tokens, non-string
    entries and a zero/negative-span cue are all filtered — and the fully empty
    input (the degenerate case the original bug turned into DER 1.0) is pinned to
    return no segments rather than raising.
    """
    # "a b c" -> non-float; None/42 -> non-string; "1.0 0.5 spk" -> t1<=t0 dropped.
    junk = ["", "not a segment line", "a b c", None, 42, "1.0 0.5 spk"]
    assert _parse_sortformer_segments(junk) == []
    # Empty backend output → empty diarization, explicitly (the DER-1.0 bug).
    assert _parse_sortformer_segments([]) == []
