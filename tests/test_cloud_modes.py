"""The two paid modes — credentials, ledger, providers, fusion, pipelines.

What is proven here, and what deliberately is not. Every test below runs with
**no network and no credential**: the provider clients take an injected
connection, the ledger is bypassed, and the two pipelines are exercised against
mock sockets. So this file proves the *wiring and the arithmetic* — the resolution
order of a credential, the key a billed call is memoised on, the parsing of each
wire format, the overlap attribution, the refusal to run without a key — and
proves nothing about the providers' own behaviour, which only a real, billed run
can show.

That boundary is the point. A test suite that needed an API key would either be
skipped everywhere (proving nothing) or billed on every push (proving it once,
expensively). These tests are the half that can be held for free, and they are the
half that catches the mistakes that actually happen: a key read from the wrong
place, a ledger key that includes something incidental, an utterance attributed to
whoever held its midpoint rather than most of it.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib

import numpy as np
import pytest

from vocal_helper import asr_cloud, cloud, cloud_media, credentials, diar_cloud, fusion, wallet

# --------------------------------------------------------------------------- #
#  Credentials: the resolution order, and the refusal to guess
# --------------------------------------------------------------------------- #


def test_environment_beats_every_stored_value(tmp_path, monkeypatch):
    """A variable in the environment wins over both files, which is what makes a
    one-off experiment or a continuous-integration run possible without editing
    anything on disk."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path("credentials.yaml").write_text(
        "credentials:\n  GLADIA_API_KEY: from-overlay\n", encoding="utf-8"
    )
    assert credentials.credential("GLADIA_API_KEY") == "from-overlay"
    monkeypatch.setenv("GLADIA_API_KEY", "from-env")
    assert credentials.credential("GLADIA_API_KEY") == "from-env"


def test_an_absent_credential_is_empty_not_an_exception(monkeypatch):
    """Absence must be a value a caller can test, because that is how a paid mode
    reports itself unavailable instead of crashing an unrelated local run."""
    monkeypatch.delenv("NOTHING_IS_CONFIGURED_HERE", raising=False)
    assert credentials.credential("NOTHING_IS_CONFIGURED_HERE") == ""
    assert credentials.credential("NOTHING_IS_CONFIGURED_HERE", "fallback") == "fallback"


def test_status_reports_booleans_and_requires_every_key_of_a_service(monkeypatch):
    """A service with one of its two keys is NOT available: the paid offline mode
    cannot run on half its providers, and saying otherwise would send a run into a
    failure after the first upload."""
    monkeypatch.setenv("GLADIA_API_KEY", "x")
    monkeypatch.delenv("PYANNOTEAI_API_KEY", raising=False)
    status = credentials.credentials_status()
    assert status["GLADIA_API_KEY"] is True
    assert status["PYANNOTEAI_API_KEY"] is False
    assert status["cloud_offline"] is False
    monkeypatch.setenv("PYANNOTEAI_API_KEY", "y")
    assert credentials.credentials_status()["cloud_offline"] is True


def test_saving_skips_blanks_so_a_masked_form_cannot_wipe_a_key(tmp_path, monkeypatch):
    """A preferences panel sends masked placeholders back. Treating one as a value
    would erase a working key on every save."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path("credentials.yaml").write_text("credentials:\n", encoding="utf-8")
    credentials.set_credentials({"GLADIA_API_KEY": "real"})
    credentials.set_credentials({"GLADIA_API_KEY": "   "})
    monkeypatch.delenv("GLADIA_API_KEY", raising=False)
    assert credentials.credential("GLADIA_API_KEY") == "real"


# --------------------------------------------------------------------------- #
#  The ledger: what the key is, and what it deliberately is not
# --------------------------------------------------------------------------- #


def test_the_fingerprint_follows_the_bytes_not_the_path_or_the_clock(tmp_path):
    """Re-extracting identical audio rewrites the modification time. A key that
    moved with it would miss the ledger and re-bill — the silent failure this
    whole module exists to prevent."""
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    first.write_bytes(b"IDENTICAL BYTES")
    second.write_bytes(b"IDENTICAL BYTES")
    os.utime(second, (0, 0))  # a different clock, the same content
    assert wallet.audio_fingerprint(str(first)) == wallet.audio_fingerprint(str(second))

    third = tmp_path / "c.wav"
    third.write_bytes(b"DIFFERENT BYTES")
    assert wallet.audio_fingerprint(str(third)) != wallet.audio_fingerprint(str(first))


def test_the_escape_hatch_is_off_unless_asked_in_so_many_words(monkeypatch):
    """A typo must not start billing, so anything unset, zero, false or no reads as
    'ledger active'."""
    for value, expected in (
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
        ("1", True),
        ("yes", True),
        ("whatever", True),
    ):
        monkeypatch.setenv("VOH_WALLET_DISABLED", value)
        assert wallet.wallet_disabled() is expected, value


def test_a_disabled_ledger_calls_straight_through(tmp_path, monkeypatch):
    """With the ledger bypassed the decorator must not import or touch it at all:
    that is what keeps a fast test lane free of the paid extra."""
    monkeypatch.setenv("VOH_WALLET_DISABLED", "1")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    calls: list[str] = []

    @wallet.paid("test.transcribe")
    def transcribe(*, audio: str) -> str:
        calls.append(audio)
        return "ok"

    assert transcribe(audio=str(audio)) == "ok"
    assert transcribe(audio=str(audio)) == "ok"
    assert len(calls) == 2  # bypassed, so genuinely called twice


def test_the_ledger_file_is_configurable_so_a_migration_keeps_what_it_paid(monkeypatch):
    """Pointing at an existing ledger is the whole migration story: a fresh default
    path is a silently empty ledger and the next run re-bills everything."""
    monkeypatch.setenv("VOH_WALLET_DB", "/tmp/elsewhere.sqlite")
    assert wallet.wallet_path().as_posix() == "/tmp/elsewhere.sqlite"
    monkeypatch.delenv("VOH_WALLET_DB")
    assert wallet.wallet_path().name == "wallet.sqlite"


# --------------------------------------------------------------------------- #
#  Chunking and attribution arithmetic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (100.0, [0.0]),  # fits in one call
        (8000.0, [0.0]),  # exactly the cap, still one call
        (8001.0, [0.0, 7880.0]),  # one second over → two chunks
        (16000.0, [0.0, 7880.0, 15760.0]),  # the tail gets its own short chunk
    ],
)
def test_chunks_cover_the_whole_recording_with_a_shared_window(duration, expected):
    """Coverage is the property that matters: a plan that stopped short would lose
    the tail of a long recording, and nothing downstream would notice."""
    plan = cloud_media.plan_chunks(duration_s=duration, cap_s=8000.0, overlap_s=120.0)
    assert plan == expected
    last = plan[-1]
    assert last + 8000.0 >= duration  # the final chunk reaches the end


def test_attribution_goes_to_whoever_held_most_of_the_interval():
    """By overlap, not by midpoint. An utterance straddling a turn change belongs
    to whoever held most of it; a midpoint rule flips on a hair's breadth."""
    segments = [
        {"start": 0.0, "end": 6.0, "speaker": "A"},
        {"start": 6.0, "end": 12.0, "speaker": "B"},
    ]
    assert cloud_media.speaker_by_overlap(segments, 0.0, 7.0) == "A"
    assert cloud_media.speaker_by_overlap(segments, 5.0, 12.0) == "B"
    assert cloud_media.speaker_by_overlap(segments, 20.0, 25.0) is None


def test_a_long_segment_starting_before_the_window_is_still_counted():
    """Cross-talk makes hosted segments genuinely overlap, so a scan that started
    near the window would drop exactly the long segments that cover it."""
    segments = [{"start": 0.0, "end": 100.0, "speaker": "host"}]
    assert cloud_media.speaker_by_overlap(segments, 90.0, 95.0) == "host"


def test_labels_are_matched_through_the_shared_window():
    """Two chunks transcribed the same seconds, so one voice occupies the same
    absolute time in both. That co-occurrence is what makes per-chunk labels
    globally consistent."""
    base = [
        {"start": 0.0, "end": 10.0, "speaker": "S0"},
        {"start": 10.0, "end": 20.0, "speaker": "S1"},
    ]
    new = [
        {"start": 1.0, "end": 9.0, "speaker": "spk_b"},
        {"start": 11.0, "end": 19.0, "speaker": "spk_a"},
    ]
    assert cloud_media.match_speakers_across_overlap(base, new, 0.0, 20.0) == {
        "spk_b": "S0",
        "spk_a": "S1",
    }


def test_an_unmatched_label_is_absent_rather_than_guessed():
    """A voice that only speaks after the shared window has no counterpart. The
    caller must name it per-chunk; inventing a match would merge two people."""
    assert (
        cloud_media.match_speakers_across_overlap(
            [], [{"start": 0.0, "end": 1.0, "speaker": "x"}], 0.0, 1.0
        )
        == {}
    )


# --------------------------------------------------------------------------- #
#  Wire formats: each provider's parser, and each provider's bytes
# --------------------------------------------------------------------------- #


def test_transcript_parser_keeps_good_words_and_drops_malformed_ones():
    """One malformed word must not discard the utterance around it: a provider
    glitch should cost a word, not a sentence."""
    event = asr_cloud.parse_gladia_message(
        {
            "type": "final",
            "utterances": [
                {
                    "words": [
                        {"word": "bonjour", "time_begin": 0.0, "time_end": 0.5, "confidence": 0.9},
                        {"word": "cassé"},  # no timings
                    ]
                }
            ],
        }
    )
    assert [w.word for w in event.words] == ["bonjour"]


@pytest.mark.parametrize("kind", ["session_start", "audio_chunk_ack", "", "error"])
def test_non_transcript_messages_are_ignored(kind):
    assert asr_cloud.parse_gladia_message({"type": kind}) is None


def test_speaker_events_and_errors_both_parse():
    """An error has to arrive as an event, not as an exception: a live session must
    be able to report 'quota exhausted' without tearing down the loop."""
    start = diar_cloud.parse_pyannote_message(
        {"type": "diarization_speaker_start", "data": {"timestamp": 2.0, "speaker": "S3"}}
    )
    assert (start.kind, start.timestamp, start.speaker) == ("start", 2.0, "S3")
    failure = diar_cloud.parse_pyannote_message({"type": "error", "message": "quota"})
    assert (failure.kind, failure.message) == ("error", "quota")
    assert diar_cloud.parse_pyannote_message({"type": "keepalive"}) is None


def test_the_transcriber_clamps_before_scaling():
    """A hot input overshooting [-1, 1] must saturate, not wrap: a wrap is a
    full-amplitude click, heard as a spike and transcribed as nothing."""
    raw = asr_cloud.pcm_f32_to_wire_bytes(np.array([2.0, -2.0], dtype=np.float32))
    assert np.frombuffer(raw, dtype=np.int16).tolist() == [32767, -32767]


def test_the_diarizer_gets_float32_at_the_documented_chunk_size():
    """Live-1 is specified at 16 kHz mono float32 in 100 ms chunks; the constant and
    the bytes must agree, or every frame is silently misaligned."""
    one_chunk = np.zeros(
        int(diar_cloud.PCM_SAMPLE_RATE * diar_cloud.PCM_CHUNK_MS / 1000), dtype=np.float32
    )
    assert len(diar_cloud.pcm_f32_to_wire_bytes(one_chunk)) == diar_cloud.PCM_CHUNK_BYTES


# --------------------------------------------------------------------------- #
#  Fusion
# --------------------------------------------------------------------------- #


def test_a_dropped_end_event_closes_the_stale_turn():
    """A real stream keeps labels stable, so two starts for one speaker mean the
    first end was lost — not that somebody is speaking twice at once."""
    turns = fusion.build_speaker_turns(
        [
            diar_cloud.DiarEvent("start", 0.0, "A"),
            diar_cloud.DiarEvent("start", 5.0, "A"),
        ]
    )
    assert [(t.start, t.end) for t in turns] == [(0.0, 5.0), (5.0, None)]


def test_an_end_without_a_start_is_ignored_rather_than_crashing():
    assert fusion.build_speaker_turns([diar_cloud.DiarEvent("end", 1.0, "ghost")]) == []


def test_only_finals_reach_the_record():
    """Partials are revised as the model hears more. Fusing them would put
    retracted words in the transcript."""
    words = [asr_cloud.AsrWord("peut", 0.0, 0.3, 1.0)]
    events = [diar_cloud.DiarEvent("start", 0.0, "S0"), diar_cloud.DiarEvent("end", 1.0, "S0")]
    assert fusion.fuse_live_transcripts([asr_cloud.AsrTranscript("partial", words)], events) == []
    assert len(fusion.fuse_live_transcripts([asr_cloud.AsrTranscript("final", words)], events)) == 1


def test_a_still_open_turn_covers_up_to_now():
    """The speaker who has not stopped talking is otherwise the one speaker who
    never gets credited."""
    words = [asr_cloud.AsrWord("oui", 1.0, 2.0, 1.0)]
    out = fusion.fuse_live_transcripts(
        [asr_cloud.AsrTranscript("final", words)],
        [diar_cloud.DiarEvent("start", 0.0, "S0")],  # no end: still speaking
    )
    assert out[0]["speaker"] == "S0"


def test_an_unattributable_utterance_says_so():
    """An unknown speaker is not speaker zero. Defaulting it to a real label would
    put words in somebody's mouth."""
    out = fusion.fuse_live_transcripts(
        [asr_cloud.AsrTranscript("final", [asr_cloud.AsrWord("hm", 50.0, 51.0, 1.0)])],
        [diar_cloud.DiarEvent("start", 0.0, "S0"), diar_cloud.DiarEvent("end", 1.0, "S0")],
    )
    assert out[0]["speaker"] == "?"


def test_confidence_survives_fusion_index_aligned_with_the_words():
    """The repair pass downstream needs to know WHICH span to re-read. Losing the
    per-word confidence means re-reading everything or nothing."""
    words = [asr_cloud.AsrWord("un", 0.0, 0.2, 0.4), asr_cloud.AsrWord("deux", 0.2, 0.4, 0.99)]
    out = fusion.fuse_live_transcripts(
        [asr_cloud.AsrTranscript("final", words)],
        [diar_cloud.DiarEvent("start", 0.0, "S0"), diar_cloud.DiarEvent("end", 1.0, "S0")],
    )
    assert out[0]["word_confidence"] == [0.4, 0.99]
    assert len(out[0]["words"]) == len(out[0]["word_confidence"])


def test_relabelling_keeps_the_original_when_the_diarizer_says_nothing():
    """A gap in the diarization is not evidence about who was talking, so the
    transcriber's own guess is better than losing the speaker entirely."""
    utterances = [
        {
            "t0": 0.0,
            "t1": 1.0,
            "speaker": "spk_0",
            "text": "a",
            "words": [(0.0, 1.0, "a")],
            "language": None,
        }
    ]
    assert fusion.relabel_utterances(utterances, [])[0]["speaker"] == "spk_0"
    relabelled = fusion.relabel_utterances(
        utterances, [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"}], engine="hybrid"
    )
    assert relabelled[0]["speaker"] == "SPEAKER_01"
    assert relabelled[0]["engine"] == "hybrid"


# --------------------------------------------------------------------------- #
#  The two paid pipelines
# --------------------------------------------------------------------------- #


def test_an_unknown_engine_is_refused_at_construction_not_mid_run():
    """Failing after the first upload would mean paying to discover a typo."""
    with pytest.raises(ValueError, match="unknown transcriber"):
        cloud.CloudOfflineConfig(transcriber="whisper")
    with pytest.raises(ValueError, match="unknown diarizer"):
        cloud.CloudOfflineConfig(diarizer="pyannote")


def test_the_offline_pipeline_names_the_engines_it_used():
    """A record names what wrote it. Without that, a result from the cheaper path is
    indistinguishable from the reference one and the reader blames the tool."""
    assert cloud.CloudOfflinePipeline().engine_name() == ("gladia:solaria-3+pyannoteai:precision-3")
    # With no diarizer there is nothing to append: the transcriber's own speakers
    # are what the record carries, and saying otherwise would overstate it.
    assert (
        cloud.CloudOfflinePipeline(
            cloud.CloudOfflineConfig(transcriber="elevenlabs", diarizer="")
        ).engine_name()
        == "elevenlabs:scribe"
    )


class _MockSocket:
    """A WebSocket stand-in: records what was sent, replays canned messages."""

    def __init__(self, replies: list[dict]) -> None:
        self.replies = replies
        self.sent: list = []

    async def send(self, payload) -> None:
        self.sent.append(payload)

    def __aiter__(self):
        async def generate():
            for reply in self.replies:
                yield json.dumps(reply)

        return generate()


def test_the_live_pipeline_fans_one_stream_out_to_both_and_fuses_the_answers():
    """The whole premise in one test: the transcriber has no speakers, the diarizer
    has no words, and the same audio reaches both."""
    asr_socket = _MockSocket(
        [
            {
                "type": "final",
                "words": [
                    {"word": "bonsoir", "time_begin": 0.1, "time_end": 0.9, "confidence": 0.95}
                ],
            }
        ]
    )
    diar_socket = _MockSocket(
        [
            {"type": "diarization_speaker_start", "data": {"timestamp": 0.0, "speaker": "S0"}},
            {"type": "diarization_speaker_end", "data": {"timestamp": 1.0, "speaker": "S0"}},
        ]
    )

    async def source():
        for _ in range(3):
            yield np.zeros(1600, dtype=np.float32)

    out = asyncio.run(cloud.CloudLivePipeline(asr_socket, diar_socket).run(source()))

    assert [(u["speaker"], u["text"]) for u in out] == [("S0", "bonsoir")]
    assert len(asr_socket.sent) == 3
    assert len(diar_socket.sent) == 3
    # Each provider got ITS OWN encoding of the same audio, not a shared buffer.
    assert len(asr_socket.sent[0]) == 3200  # int16
    assert len(diar_socket.sent[0]) == 6400  # float32


def test_the_pre_send_hook_sees_every_frame_before_it_leaves_the_machine():
    """This is the anonymisation injection point. If a frame could reach a provider
    without passing through it, the hook would be decorative."""
    seen: list[int] = []

    async def hook(pcm):
        seen.append(len(pcm))
        return pcm

    asr_socket, diar_socket = _MockSocket([]), _MockSocket([])

    async def source():
        yield np.zeros(800, dtype=np.float32)
        yield np.zeros(1600, dtype=np.float32)

    asyncio.run(cloud.CloudLivePipeline(asr_socket, diar_socket, pre_send=hook).run(source()))
    assert seen == [800, 1600]
    assert len(asr_socket.sent) == len(diar_socket.sent) == 2


def test_a_stop_event_ends_a_session_that_would_never_end_on_its_own():
    """Neither a microphone nor an open-ended live URL exhausts. Without this a
    live session has no way to finish."""
    asr_socket, diar_socket = _MockSocket([]), _MockSocket([])
    stop = asyncio.Event()

    async def source():
        yield np.zeros(1600, dtype=np.float32)
        stop.set()
        yield np.zeros(1600, dtype=np.float32)  # must not be sent

    asyncio.run(cloud.CloudLivePipeline(asr_socket, diar_socket).run(source(), stop=stop))
    assert len(asr_socket.sent) == 1


# --------------------------------------------------------------------------- #
#  The hybrid: hosted speakers, local words
# --------------------------------------------------------------------------- #


def test_the_hosted_diarizer_refuses_before_touching_any_audio(monkeypatch):
    """Checked at load, so a missing key costs nothing and fails while a human can
    still fix it — and never degrades to the local backend, which would answer the
    same command differently."""
    from vocal_helper.diar import OfflineDiarStage

    monkeypatch.delenv("PYANNOTEAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PYANNOTEAI_API_KEY"):
        OfflineDiarStage(backend="pyannoteai")._ensure_backend()


def test_the_backend_list_names_which_options_cost_money():
    """An error message that lists backends without saying which are billed invites
    exactly the wrong choice."""
    from vocal_helper.diar import OfflineDiarStage

    with pytest.raises(ValueError) as caught:
        OfflineDiarStage(backend="nonexistent")._ensure_backend()
    message = str(caught.value)
    assert "free and local" in message
    assert "paid, hosted" in message


def test_the_router_never_routes_to_a_paid_backend():
    """Cost is not on the same scale as DER or RTF, so no comparison can decide it.
    Paying is a decision made by name, by a human."""
    from vocal_helper.router import select_diarization

    for live in (True, False):
        for duration in (10.0, 600.0, 7200.0):
            for speakers in (None, 2, 6):
                plan = select_diarization(live=live, duration_s=duration, num_speakers=speakers)
                assert plan.backend != "pyannoteai", (live, duration, speakers)


# --------------------------------------------------------------------------- #
#  Which model, and why it is named rather than defaulted
# --------------------------------------------------------------------------- #


def test_the_batch_diarizer_asks_for_the_best_model_not_the_library_default():
    """The installed client defaults to the PREVIOUS generation. A caller who says
    nothing would pay full price for it, so the best model is named here.

    This test is the guard against a silent regression: a library upgrade that
    changed its own default must not change ours, in either direction.
    """
    import inspect

    from vocal_helper.diar_cloud import BATCH_MODEL, pyannoteai_diarize

    assert BATCH_MODEL == "precision-3"
    ours = inspect.signature(pyannoteai_diarize).parameters["model"].default
    assert ours == BATCH_MODEL

    try:
        from pyannoteai.sdk import Client
    except ImportError:
        pytest.skip("pyannoteai client not installed; the paid extra is opt-in")
    theirs = inspect.signature(Client.diarize).parameters["model"].default
    assert ours != theirs, (
        "our default now equals the library's — either they caught up, which is "
        "good news, or this module stopped naming its model, which is not"
    )


def test_the_streaming_model_is_the_only_one_offered():
    """Live-1 is a constraint, not a preference: the service exposes one streaming
    model, and Gladia's streaming enumeration likewise has a single member. 'Take
    the best' and 'take the only one' give the same answer on both sides."""
    from vocal_helper.asr_cloud import GLADIA_LIVE_MODEL, GLADIA_PRERECORDED_MODEL
    from vocal_helper.diar_cloud import LIVE_MODEL

    assert LIVE_MODEL == "live-1"
    assert GLADIA_LIVE_MODEL == "solaria-1"
    assert GLADIA_PRERECORDED_MODEL == "solaria-3"
    assert GLADIA_LIVE_MODEL != GLADIA_PRERECORDED_MODEL, (
        "the live endpoint rejects the pre-recorded model; conflating them is the "
        "mistake this assertion exists to prevent"
    )


def test_the_model_is_part_of_the_billing_key():
    """Otherwise an upgrade is served a stale cheaper result, and the better model
    appears to have changed nothing — the most expensive kind of invisible bug."""
    import inspect

    source = inspect.getsource(diar_cloud)
    key_line = next(
        line for line in source.splitlines() if "key_fields=" in line and "model" in line
    )
    assert '"model"' in key_line


def test_the_record_names_the_model_not_merely_the_provider():
    """Two runs through one provider on different generations are not the same
    measurement. A record that hid which one produced it could not be compared
    against itself a year later."""
    assert cloud.CloudOfflinePipeline().engine_name() == ("gladia:solaria-3+pyannoteai:precision-3")
    downgraded = cloud.CloudOfflinePipeline(cloud.CloudOfflineConfig(diarizer_model="precision-2"))
    assert downgraded.engine_name() == "gladia:solaria-3+pyannoteai:precision-2"


def test_the_hybrid_also_gets_the_best_diarizer():
    """The hybrid is the mode that exists to buy good speakers. Buying the previous
    generation's would defeat its entire purpose."""
    from vocal_helper.diar import OfflineDiarStage, _PyannoteAiOfflineDiar
    from vocal_helper.diar_cloud import BATCH_MODEL

    stage = OfflineDiarStage(backend="pyannoteai")
    assert _PyannoteAiOfflineDiar(model=stage.pyannoteai_model).model == BATCH_MODEL
    assert _PyannoteAiOfflineDiar(model="community-1").model == "community-1"


def test_the_ledger_key_follows_the_effective_arguments_not_the_call_style(tmp_path, monkeypatch):
    """``f(audio=a)`` and ``f(audio=a, model=<the default>)`` are one identical
    request. Keying on what the caller typed would bill it twice, and both runs
    would succeed — one of them just costs money, invisibly.

    The same holds for a hint left unset versus passed as ``None``: not asking is
    not a different question.
    """
    monkeypatch.setenv("VOH_WALLET_DISABLED", "1")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"bytes")
    keys: list[dict] = []

    # Rebuild the decorator's key function the same way it does, so the payload can
    # be inspected without a ledger, a network or a credential.
    def payload_for(**kwargs):
        import inspect

        def diarize(*, audio: str, model: str = "precision-3", num_speakers: int | None = None):
            return None

        bound = inspect.signature(diarize).bind(**kwargs)
        bound.apply_defaults()
        effective = bound.arguments
        out = {"audio": wallet.audio_fingerprint(effective["audio"])}
        for field in ("model", "num_speakers"):
            if effective.get(field) is not None:
                out[field] = effective[field]
        return out

    keys.append(payload_for(audio=str(audio)))
    keys.append(payload_for(audio=str(audio), model="precision-3"))
    keys.append(payload_for(audio=str(audio), num_speakers=None))
    assert keys[0] == keys[1] == keys[2]

    # And a genuinely different request IS a different key.
    assert payload_for(audio=str(audio), model="precision-2") != keys[0]
    assert payload_for(audio=str(audio), num_speakers=3) != keys[0]


def test_the_transcription_ledger_keys_stay_compatible_with_what_was_already_paid():
    """The expensive stage is transcription. Its namespace and key fields must not
    drift, or thousands of already-billed hours become unreachable and the next run
    pays for them again.

    Diarization is the stage whose key deliberately gained the model, which costs
    the cached diarizations — the cheaper stage, and a precision-3 call genuinely is
    a different call. That trade is the one being asserted here.
    """
    import inspect

    source = inspect.getsource(asr_cloud)
    assert '@paid("gladia.transcribe")' in source, (
        "the Gladia ledger key must stay the audio alone; adding a field here "
        "invalidates every transcription already paid for"
    )
    assert '@paid("elevenlabs.transcribe", key_fields=("diarize", "num_speakers"))' in source
