"""
Tests for the speaker-count hints on :class:`vocal_helper.diar.OfflineDiarStage`.

Module summary
--------------
A known roster has to reach the backend's **own** clustering, not be applied as a repair
afterwards. That is not a style preference: ``pdbms`` 685b9cc measured that an
over-estimated speaker count cannot be merged back down with the same embedding that
produced it (AUC 0.60 median, 0.27 worst — at or below chance), because the clusterer has
already spent exactly that information. A count hint is only worth anything while it can
still change the decision.

Model-free throughout: the backends are stubs, so no ONNX, torch or model files are touched.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import numpy as np
import pytest

from vocal_helper.diar import OfflineDiarStage, _count_kwargs


class _RecordingBackend:
    """A backend that records the hints it was handed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def load(self) -> None:
        """No-op."""

    def diarize(self, pcm, sr, *, num_speakers=None, min_speakers=None, max_speakers=None):
        """Record the hints and return one fixed turn."""
        self.calls.append(
            {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }
        )
        return [(0.0, 1.0, "spk0")]


class _OldBackend:
    """A backend predating the hints — it must still work, not explode."""

    def load(self) -> None:
        """No-op."""

    def diarize(self, pcm, sr):
        """Accept only the positional contract."""
        return [(0.0, 1.0, "spk0")]


def _stage(backend_obj) -> OfflineDiarStage:
    stage = OfflineDiarStage(backend="sherpa")
    stage._backend_obj = backend_obj
    stage._embedder = object()
    return stage


_PCM = np.zeros(16_000, dtype=np.float32)


# --------------------------------------------------------------- the kwargs policy


def test_a_floor_is_preferred_over_an_exact_count():
    """A roster proves people are distinct; it does not prove nobody else speaks.

    Panel episodes also carry reporters in packages, archive clips and off-screen
    narrators. Forcing an exact count has to put those voices somewhere, and where it puts
    them is inside a real person.
    """
    assert _count_kwargs(None, 6, None) == {"min_speakers": 6}


def test_an_explicit_exact_count_wins_when_the_caller_asserts_one():
    assert _count_kwargs(4, 6, 9) == {"num_speakers": 4}


def test_no_hints_means_no_kwargs_so_auto_detection_is_untouched():
    assert _count_kwargs(None, None, None) == {}


def test_nonsense_counts_are_dropped_rather_than_forwarded():
    """Zero or negative speakers is not a bound, it is a bug upstream."""
    assert _count_kwargs(0, 0, 0) == {}
    assert _count_kwargs(None, -3, None) == {}


# --------------------------------------------------------------- stage plumbing


def test_the_stage_forwards_hints_to_the_backend():
    backend = _RecordingBackend()
    _stage(backend).diarize(_PCM, 16_000, min_speakers=6)
    assert backend.calls == [{"num_speakers": None, "min_speakers": 6, "max_speakers": None}]


def test_the_stage_passes_nothing_when_nothing_was_asked():
    backend = _RecordingBackend()
    _stage(backend).diarize(_PCM, 16_000)
    assert backend.calls == [{"num_speakers": None, "min_speakers": None, "max_speakers": None}]


def test_a_backend_without_hints_still_runs():
    """NeMo Sortformer fixes its own speaker slots and takes no hint.

    Degrading to a plain call is right; failing on an argument that backend never had would
    make a hint nobody needed break a pipeline that worked.
    """
    assert _stage(_OldBackend()).diarize(_PCM, 16_000, min_speakers=6) == [(0.0, 1.0, "spk0")]


def test_a_backend_without_hints_warns_rather_than_silently_ignoring(caplog):
    """Silently dropping a roster is the failure that looks like success."""
    with caplog.at_level("WARNING"):
        _stage(_OldBackend()).diarize(_PCM, 16_000, min_speakers=6)
    assert "ignores speaker-count hints" in caplog.text


def test_no_warning_when_there_was_no_hint_to_ignore(caplog):
    with caplog.at_level("WARNING"):
        _stage(_OldBackend()).diarize(_PCM, 16_000)
    assert "ignores speaker-count hints" not in caplog.text


# --------------------------------------------------------------- the sherpa floor


class _FakeSherpa:
    """Stands in for sherpa's clustering: returns `configured` speakers, or 2 on auto."""

    def __init__(self) -> None:
        self.builds: list[int] = []
        self._n = 2

    def _reconfigure(self, n: int) -> None:
        self.builds.append(n)
        self._n = n
        self._configured_clusters = n

    _configured_clusters = -1
    num_clusters = -1

    def load(self) -> None:
        """No-op."""

    def diarize(self, pcm, sr, *, num_speakers=None, min_speakers=None, max_speakers=None):
        """Mimic _SherpaOfflineDiar.diarize's floor logic over a fake clusterer."""
        from vocal_helper.diar import _SherpaOfflineDiar

        self._sd = object()
        return _SherpaOfflineDiar.diarize(
            self,
            pcm,
            sr,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    def _run_fake(self):
        return [(float(i), float(i) + 1.0, f"spk{i}") for i in range(self._n)]


def test_sherpa_floor_triggers_a_second_pass_only_when_it_under_counted(monkeypatch):
    """The floor costs a rerun exactly when the first pass found too few people.

    Too few speakers is the unrecoverable direction — two people inside one label cannot be
    separated later — so paying one extra pass to avoid it is the right trade.
    """
    from vocal_helper.diar import _SherpaOfflineDiar

    fake = _FakeSherpa()
    monkeypatch.setattr(
        _SherpaOfflineDiar,
        "_reconfigure",
        lambda self, n: _FakeSherpa._reconfigure(fake, n),
        raising=False,
    )

    diar = _SherpaOfflineDiar()
    diar._sd = fake
    diar._configured_clusters = -1
    monkeypatch.setattr(
        type(fake), "process", lambda self, pcm: _Result(fake._run_fake()), raising=False
    )

    # auto pass finds 2; a floor of 5 must force a rebuild at 5
    out = diar.diarize(_PCM, 16_000, min_speakers=5)
    assert fake.builds == [5]
    assert len({lab for _a, _b, lab in out}) == 5

    # and with the floor already satisfied, no rebuild happens
    fake.builds.clear()
    diar.diarize(_PCM, 16_000, min_speakers=2)
    assert fake.builds == []


class _Result:
    def __init__(self, segs):
        self._segs = [_Seg(a, b, lab) for a, b, lab in segs]

    def sort_by_start_time(self):
        return self._segs


class _Seg:
    def __init__(self, a, b, lab):
        self.start, self.end, self.speaker = a, b, lab.replace("spk", "")


def test_pyannote_backend_accepts_the_hints():
    """The gap this closed: the pyannote backend took no count hint at all."""
    import inspect

    from vocal_helper.diar import _PyannoteOfflineDiar

    sig = inspect.signature(_PyannoteOfflineDiar.diarize).parameters
    for name in ("num_speakers", "min_speakers", "max_speakers"):
        assert sig[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig[name].default is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
