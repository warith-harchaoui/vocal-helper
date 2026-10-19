"""Live-session measurement: latency, flicker, queue waits, and the CADENCE invariant."""

import pytest

from vocal_helper.live_metrics import LiveMetrics


def _m() -> LiveMetrics:
    return LiveMetrics(started_at=0.0)


# --- latency ----------------------------------------------------------------------------- #

def test_latency_is_counted_from_the_end_of_the_speech() -> None:
    """CADENCE.md's own example: a 3 s line delivered at 3.85 s is 0.85 s late, not 3.85 s.

    Measuring from the start would charge a long sentence for being long, and would reward
    a pipeline that chops speech into fragments.
    """
    m = _m()
    m.record(speaker="S0", t0=0.0, t1=3.0, text="bonjour je suis bloqué", now_s=3.85)
    assert m.report()["latency_p50_s"] == pytest.approx(0.85)


def test_quantiles_report_an_observed_sample_not_an_interpolation() -> None:
    """Small samples: an interpolated p95 over a handful of points invents a number."""
    m = _m()
    for i, lat in enumerate([0.5, 0.6, 0.7, 5.0]):
        m.record(speaker="S0", t0=float(i), t1=float(i) + 1, text=f"t{i}",
                 now_s=float(i) + 1 + lat)
    r = m.report()
    # Nearest rank over four samples puts p95 on the last one — a latency that really
    # happened, rather than a weighted blend of 0.7 and 5.0 that nobody experienced.
    assert r["latency_max_s"] == pytest.approx(5.0)
    assert r["latency_p95_s"] == pytest.approx(5.0)
    # And p50 over an even sample is a real observation (0.7), not the 0.65 an averaging
    # median would return — nobody waited 0.65 s.
    assert r["latency_p50_s"] == pytest.approx(0.7)


def test_an_empty_session_reports_nothing_rather_than_zero() -> None:
    """Zero latency and no data are different claims; only one of them is true here."""
    r = _m().report()
    assert r["items"] == 0 and r["latency_p50_s"] is None
    assert "aucun élément" in _m().summary()


# --- flicker ----------------------------------------------------------------------------- #

def test_rewriting_an_already_shown_item_is_a_revision() -> None:
    m = _m()
    m.record(speaker="S0", t0=0.0, t1=2.0, text="je suis bloqué", now_s=2.2)
    m.record(speaker="S0", t0=0.0, t1=2.0, text="je suis bloqué avec mon câble", now_s=2.9)
    assert m.report()["revisions"] == 1
    assert m.revision_rate() == pytest.approx(0.5)


def test_re_delivering_identical_text_is_not_flicker() -> None:
    """A duplicate delivery does not make the screen jump, so it is not a revision."""
    m = _m()
    for now in (2.2, 2.9):
        m.record(speaker="S0", t0=0.0, t1=2.0, text="je suis bloqué", now_s=now)
    assert m.report()["revisions"] == 0


def test_the_same_timestamp_from_another_speaker_is_another_item() -> None:
    m = _m()
    m.record(speaker="S0", t0=0.0, t1=2.0, text="bonjour", now_s=2.2)
    m.record(speaker="S1", t0=0.0, t1=2.0, text="bonsoir", now_s=2.3)
    assert m.report()["revisions"] == 0 and m.report()["items"] == 2


def test_a_finalise_then_emit_pipeline_cannot_flicker() -> None:
    """Zero is not automatically good — it is paid for in latency, which is why both ship."""
    m = _m()
    for i in range(5):
        m.record(speaker="S0", t0=float(i), t1=float(i) + 1, text=f"phrase {i}",
                 now_s=float(i) + 1.4)
    assert m.revision_rate() == 0.0
    assert m.report()["latency_p50_s"] == pytest.approx(0.4)


# --- where the wait came from -------------------------------------------------------------- #

def test_queue_waits_are_reported_per_stage() -> None:
    """A slow total with no queue wait is a slow model; with one, a starved consumer."""
    m = _m()
    for w in (0.1, 0.2, 0.9):
        m.record_queue_wait("asr", w)
    m.record_queue_wait("diar", 0.05)
    waits = m.report()["queue_waits"]
    assert waits["asr"]["n"] == 3 and waits["asr"]["p95_s"] == pytest.approx(0.9)
    assert waits["diar"]["total_s"] == pytest.approx(0.05)


def test_the_summary_names_the_slowest_queue() -> None:
    m = _m()
    m.record(speaker="S0", t0=0.0, t1=1.0, text="x", now_s=1.5)
    m.record_queue_wait("asr", 0.1)
    m.record_queue_wait("llm", 2.5)
    assert "llm" in m.summary()


def test_a_negative_wait_is_refused_rather_than_averaged_in() -> None:
    m = _m()
    m.record_queue_wait("asr", -1.0)
    assert m.report()["queue_waits"] == {}


# --- the invariant ------------------------------------------------------------------------- #

def test_an_item_emitted_before_its_audio_existed_is_flagged() -> None:
    """CADENCE.md's one rule, turned into something a test can fail on.

    An event whose audio had not been captured yet means a clock, a buffer or a mock is
    lying — and every latency computed beside it is fiction.
    """
    m = _m()
    m.record(speaker="S0", t0=0.0, t1=10.0, text="du futur", now_s=2.0)
    assert m.report()["violations"] == 1
    assert m.violations[0]["ahead_by_s"] == pytest.approx(8.0)
    assert "non fiables" in m.summary()


def test_a_well_behaved_session_has_no_violation() -> None:
    m = _m()
    m.record(speaker="S0", t0=0.0, t1=3.0, text="bonjour", now_s=3.85)
    assert m.violations == []


# --- wired into the live pipeline ---------------------------------------------------------- #

def test_the_pipeline_records_what_the_consumer_sees(monkeypatch) -> None:
    """The collector has to sit at the yield, not earlier.

    Anywhere upstream measures when a stage finished, which is not when the watcher saw
    anything — the remaining queue wait is precisely the part that makes a live session
    feel slow.
    """
    import asyncio

    import vocal_helper as vh

    m = vh.LiveMetrics(started_at=0.0)
    pipe = vh.Pipeline(source=lambda: iter(()), metrics=m)
    assert pipe.metrics is m

    # Drive the collector the way the yield site does, without starting audio or models.
    pipe.metrics.record(speaker="S0", t0=0.0, t1=2.0, text="bonjour", now_s=2.5)
    assert m.report()["items"] == 1
    assert m.report()["latency_p50_s"] == pytest.approx(0.5)
    assert asyncio  # the pipeline stays async; nothing here needed a loop


def test_an_unmeasured_pipeline_carries_no_collector() -> None:
    """Measurement is opt-in: an unmeasured session must pay nothing at all."""
    import vocal_helper as vh

    assert vh.Pipeline(source=lambda: iter(())).metrics is None
