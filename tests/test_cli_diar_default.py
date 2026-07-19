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


@pytest.mark.parametrize(
    ("pyannote", "nemo", "duration_s", "use_offline", "diar", "note_needles"),
    [
        # Both backends + short ⇒ router's short/dense branch → offline nemo,
        # and the nudge surfaces the chosen backend plus its quality/speed.
        (True, True, SHORT_S, True, {"backend": "nemo"}, ("nemo", "DER")),
        # Duration past the 300 s ceiling ⇒ robust long-form branch → pyannote.
        (True, True, LONG_S, True, {"backend": "pyannote"}, ("pyannote",)),
        # Unknown length (probe failed) is NOT read as short → safe long branch.
        (True, True, None, True, {"backend": "pyannote"}, ("pyannote",)),
        # Short, but the nemo extra is absent ⇒ pyannote, never an unrunnable pick.
        (True, False, SHORT_S, True, {"backend": "pyannote"}, ("pyannote",)),
    ],
)
def test_auto_batch_routes_offline_by_duration(
    probes,
    pyannote: bool,
    nemo: bool,
    duration_s: float | None,
    use_offline: bool,
    diar: dict,
    note_needles: tuple[str, ...],
) -> None:
    """Auto batch selection prefers the offline whole-buffer diarizer, routed by real duration.

    The router (the *aiguilleur*) is actually *enforced* for files: with an
    offline backend installed, a batch run goes offline and picks the backend
    from the file's probed length — short/dense (≤300 s) → nemo, long / unknown
    → pyannote — degrading to pyannote when the nemo extra is missing. The nudge
    note names the chosen backend (and, for nemo, its DER/RTF).

    Parameters
    ----------
    probes : Callable[[bool, bool], None]
        Fixture pinning the pyannote / nemo availability probes.
    pyannote, nemo : bool
        Whether each offline backend is "installed" for this scenario.
    duration_s : float or None
        Probed clip length; ``None`` models a failed probe.
    use_offline : bool
        Expected offline-vs-online decision.
    diar : dict
        Expected diarizer config the chooser emits.
    note_needles : tuple of str
        Substrings the surfaced nudge note must contain.
    """
    probes(pyannote, nemo)
    got_offline, got_diar, note = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=False, duration_s=duration_s
    )
    assert got_offline is use_offline
    assert got_diar == diar
    assert note is not None
    for needle in note_needles:
        assert needle in note


def test_no_offline_backend_falls_back_to_online_refine(probes) -> None:
    """No offline backend installed → online diarizer with the batch refine pass."""
    # Neither pyannote nor nemo available ⇒ the runnable online fallback; the
    # online backend is still routed (auto → nemo), plus the refine knob.
    probes(False, False)
    use_offline, diar, note = cli._choose_file_diar(
        BASE, explicit_offline=False, batch=True, force_online=False, duration_s=SHORT_S
    )
    assert use_offline is False
    assert diar == {"backend": "nemo", "refine_on_close": True}
    assert note and "refine" in note


@pytest.mark.parametrize(
    (
        "explicit_offline",
        "force_online",
        "batch",
        "requested_backend",
        "duration_s",
        "use_offline",
        "diar",
        "note_is_none",
    ),
    [
        # ``--online`` forces streaming even with a full bundle, plus the batch
        # refine knob; an explicit online choice carries no router nudge.
        (
            False,
            True,
            True,
            "auto",
            SHORT_S,
            False,
            {"backend": "nemo", "refine_on_close": True},
            True,
        ),
        # ``--offline --diar-backend nemo`` honours the operator's backend verbatim
        # (router not consulted, probes irrelevant), even on a long file that would
        # otherwise auto-pick pyannote; an explicit backend carries no note.
        (True, False, True, "nemo", LONG_S, True, {"backend": "nemo"}, True),
        # ``--offline`` with no explicit backend still routes by duration (short →
        # nemo); this fires the router, so it *does* carry its rationale note.
        (True, False, True, "auto", SHORT_S, True, {"backend": "nemo"}, False),
        # Non-batch (real-time replay) stays online with a routed backend and no
        # batch-only refine knob; auto routing still surfaces the router note.
        (False, False, False, "auto", SHORT_S, False, {"backend": "nemo"}, False),
    ],
)
def test_explicit_flags_and_realtime_override_router(
    probes,
    explicit_offline: bool,
    force_online: bool,
    batch: bool,
    requested_backend: str,
    duration_s: float,
    use_offline: bool,
    diar: dict,
    note_is_none: bool,
) -> None:
    """Explicit ``--online`` / ``--offline`` / ``--diar-backend`` and real-time replay are honoured.

    Operator overrides win over the auto router: ``--online`` forces streaming
    (with the batch refine knob) regardless of the installed bundle;
    ``--offline`` pins the mode yet still routes the backend by duration unless
    an explicit ``--diar-backend`` is given (then it is used verbatim, probes
    irrelevant); a non-batch replay stays online with no batch-only refine. An
    explicit backend / forced-online choice carries no nudge, whereas an auto
    route still surfaces the router's rationale.

    Parameters
    ----------
    probes : Callable[[bool, bool], None]
        Fixture pinning the pyannote / nemo availability probes.
    explicit_offline, force_online, batch : bool
        The ``--offline`` / ``--online`` / batch-mode flags under test.
    requested_backend : str
        An explicit ``--diar-backend``, or ``"auto"`` for router routing.
    duration_s : float
        Probed clip length driving the auto routes.
    use_offline : bool
        Expected offline-vs-online decision.
    diar : dict
        Expected diarizer config the chooser emits.
    note_is_none : bool
        Whether the chooser must suppress the router nudge for this override.
    """
    # Both backends installed: the explicit-backend row proves the override wins
    # even though the router would otherwise pick differently for that duration.
    probes(True, True)
    got_offline, got_diar, note = cli._choose_file_diar(
        BASE,
        explicit_offline=explicit_offline,
        batch=batch,
        force_online=force_online,
        duration_s=duration_s,
        requested_backend=requested_backend,
    )
    assert got_offline is use_offline
    assert got_diar == diar
    assert (note is None) is note_is_none


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
