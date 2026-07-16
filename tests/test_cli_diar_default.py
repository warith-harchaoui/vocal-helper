"""
Tests for the batch-file diarization default (``_choose_file_diar``).

Pins the reliability policy established by the 2026-07-16 DER sweep (offline
pyannote DER ~0.12 on AMI vs ~0.50 for the online diarizer): a batch file run
should prefer the offline pyannote diarizer when its bundle is present, fall
back to the online diarizer *with the refine pass* otherwise, and honour the
explicit ``--offline`` / ``--online`` overrides. Model-free — the bundle
availability probe is monkeypatched.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import pytest

from vocal_helper import cli_argparse as cli

BASE = {"backend": "nemo"}


@pytest.fixture
def bundle(monkeypatch):
    """Factory to set the offline-pyannote availability probe."""

    def _set(available: bool) -> None:
        monkeypatch.setattr(cli, "_offline_pyannote_available", lambda: available)

    return _set


def test_batch_prefers_offline_pyannote_when_bundle_present(bundle) -> None:
    bundle(True)
    use_offline, diar, note = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=False
    )
    assert use_offline is True
    assert diar == {"backend": "pyannote"}  # auto-upgrade to the DER-best backend
    assert note and "offline pyannote" in note


def test_batch_falls_back_to_online_refine_without_bundle(bundle) -> None:
    bundle(False)
    use_offline, diar, note = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=False
    )
    assert use_offline is False
    assert diar == {"backend": "nemo", "refine_on_close": True}
    assert note and "refine" in note


def test_online_flag_forces_streaming_even_with_bundle(bundle) -> None:
    bundle(True)
    use_offline, diar, note = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=True
    )
    assert use_offline is False
    assert diar == {"backend": "nemo", "refine_on_close": True}
    assert note is None  # explicit choice → no nudge


def test_explicit_offline_honours_backend(bundle) -> None:
    bundle(False)  # explicit --offline shouldn't depend on the probe
    use_offline, diar, note = cli._choose_file_diar(
        {"backend": "nemo"}, explicit_offline=True, batch=True, force_online=False
    )
    assert use_offline is True
    assert diar == {"backend": "nemo"}  # honours --diar-backend, no forced pyannote


def test_realtime_file_stays_online_without_refine(bundle) -> None:
    bundle(True)
    use_offline, diar, note = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=False, force_online=False
    )
    assert use_offline is False
    assert diar == {"backend": "nemo"}  # no refine when not batch
    assert "refine_on_close" not in diar


def test_choose_does_not_mutate_input(bundle) -> None:
    bundle(False)
    base = {"backend": "nemo"}
    cli._choose_file_diar(base, explicit_offline=False, batch=True, force_online=False)
    assert base == {"backend": "nemo"}  # returns a fresh dict
