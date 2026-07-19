"""
Tests for router-enforced batch-file diarization selection (``_choose_file_diar``).

Pins the contract that the study-grounded router (the *aiguilleur*) is actually
*enforced* for files, not decorative: a batch run prefers the offline
whole-buffer diarizer when a backend is installed, and the backend is chosen from
the file's **real probed duration** — short/dense (≤300 s) → ``nemo``, long /
unknown → ``pyannote`` — with a fall-back to the online diarizer + refine pass
when no offline backend can run. Explicit ``--offline`` / ``--online`` /
``--diar-backend`` overrides are honoured. Model-free: both availability probes
are monkeypatched, so nothing loads a real diarizer.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import pytest

from vocal_helper import cli_argparse as cli

# A neutral base config — the backend now flows in via ``requested_backend``,
# not via this dict, so it must never leak through as the chosen backend.
BASE: dict = {}

# Representative durations either side of the router's 300 s NeMo ceiling.
SHORT_S = 45.0
LONG_S = 1800.0


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch):
    """Factory to set both offline-backend availability probes at once.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to swap the module-level availability functions.

    Returns
    -------
    Callable[[bool, bool], None]
        ``_set(pyannote, nemo)`` — pins whether each offline backend is
        "installed" for the duration of a test, so the router's decision is
        exercised without importing pyannote / NeMo.
    """

    def _set(pyannote: bool, nemo: bool) -> None:
        """Pin the pyannote / nemo availability probes to fixed booleans."""
        monkeypatch.setattr(cli, "_offline_pyannote_available", lambda: pyannote)
        monkeypatch.setattr(cli, "_offline_nemo_available", lambda: nemo)

    return _set


def test_short_batch_routes_to_nemo(probes) -> None:
    """A short file with nemo installed routes offline to nemo (the crossover)."""
    # Both backends available + a short duration ⇒ the router's short/dense branch.
    probes(True, True)
    use_offline, diar, note = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=False, duration_s=SHORT_S
    )
    assert use_offline is True
    assert diar == {"backend": "nemo"}  # short ≤300 s → nemo, not the old constant pyannote
    assert note and "nemo" in note and "DER" in note  # quality/speed surfaced


def test_long_batch_routes_to_pyannote(probes) -> None:
    """A long file routes offline to pyannote — nemo hangs past ~25 min."""
    # A duration past the 300 s ceiling ⇒ the robust long-form branch.
    probes(True, True)
    use_offline, diar, note = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=False, duration_s=LONG_S
    )
    assert use_offline is True
    assert diar == {"backend": "pyannote"}
    assert note and "pyannote" in note


def test_unknown_duration_routes_to_pyannote(probes) -> None:
    """Unknown length (probe failed) is treated as long-form → pyannote."""
    # ``duration_s=None`` must NOT be read as short — it is the safe long branch.
    probes(True, True)
    use_offline, diar, _ = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=False, duration_s=None
    )
    assert use_offline is True
    assert diar == {"backend": "pyannote"}


def test_short_batch_without_nemo_falls_to_pyannote(probes) -> None:
    """Short file, but the nemo extra is absent → pyannote, never an unrunnable pick."""
    # nemo unavailable removes the short/dense branch; pyannote is still installed.
    probes(True, False)
    use_offline, diar, _ = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=False, duration_s=SHORT_S
    )
    assert use_offline is True
    assert diar == {"backend": "pyannote"}


def test_no_offline_backend_falls_back_to_online_refine(probes) -> None:
    """No offline backend installed → online diarizer with the refine pass."""
    # Neither pyannote nor nemo available ⇒ the runnable online fallback; the
    # online backend is still routed (auto → nemo), plus the refine knob.
    probes(False, False)
    use_offline, diar, note = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=False, duration_s=SHORT_S
    )
    assert use_offline is False
    assert diar == {"backend": "nemo", "refine_on_close": True}
    assert note and "refine" in note


def test_online_flag_forces_streaming_even_with_bundle(probes) -> None:
    """``--online`` forces the streaming diarizer regardless of installed backends."""
    # The explicit online override wins over the offline preference; no nudge.
    probes(True, True)
    use_offline, diar, note = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=True, duration_s=SHORT_S
    )
    assert use_offline is False
    assert diar == {"backend": "nemo", "refine_on_close": True}
    assert note is None  # explicit choice → no nudge


def test_explicit_offline_honours_backend_override(probes) -> None:
    """``--offline --diar-backend nemo`` honours the operator's explicit backend."""
    # An explicit backend is an override — the router is not consulted, and the
    # availability probes are irrelevant to the operator's forced choice.
    probes(False, False)
    use_offline, diar, note = cli._choose_file_diar(
        BASE,
        explicit_offline=True,
        batch=True,
        force_online=False,
        duration_s=LONG_S,
        requested_backend="nemo",
    )
    assert use_offline is True
    assert diar == {"backend": "nemo"}  # honoured verbatim, no forced pyannote
    assert note is None  # an explicit override carries no router rationale


def test_explicit_offline_auto_routes_by_duration(probes) -> None:
    """``--offline`` with no backend still routes by duration (short → nemo)."""
    # Forcing offline should not disable the router — only pin the mode to offline.
    probes(True, True)
    use_offline, diar, _ = cli._choose_file_diar(
        BASE, explicit_offline=True, batch=True, force_online=False, duration_s=SHORT_S
    )
    assert use_offline is True
    assert diar == {"backend": "nemo"}


def test_realtime_file_stays_online(probes) -> None:
    """A real-time (non-batch) file replay stays online with a routed backend."""
    # Not batch ⇒ streaming path; auto online routes to nemo, and no refine knob
    # is added (that is a batch-only over-segmentation guard).
    probes(True, True)
    use_offline, diar, _ = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=False, force_online=False, duration_s=SHORT_S
    )
    assert use_offline is False
    assert diar == {"backend": "nemo"}
    assert "refine_on_close" not in diar


def test_choose_preserves_base_keys_and_does_not_mutate(probes) -> None:
    """Base config keys (e.g. join_threshold) survive; the input dict is untouched."""
    # The chooser must merge base_diar (carrying tuned knobs) rather than replace
    # it, and must return a fresh dict so the caller's config is not mutated.
    probes(True, True)
    base = {"join_threshold": 0.25}
    _, diar, _ = cli._choose_file_diar(
        base, explicit_offline=False, batch=True, force_online=False, duration_s=SHORT_S
    )
    assert diar == {"join_threshold": 0.25, "backend": "nemo"}  # merged, not clobbered
    assert base == {"join_threshold": 0.25}  # input left intact
