"""Tests for the ``vocal-helper`` CLI surface.

Construction-only — we never drain a real pipeline, because that would need a
real source and would load the heavy stages. The goal is to assert that the
shipped click parser and the config builder do the right thing (no HuggingFace
token is involved anywhere).

The tests are organised as a handful of scenarios that each drive a realistic
slice of the CLI end-to-end — parse representative ``argv`` through the *real*
shipped commands, build the pipeline config, and assert the derived fields
together — rather than one micro-test per flag.

Parsing goes through :meth:`click.Command.make_context`, which runs the whole
option stack (``Choice``, ``Path(exists=True)``, type coercion, defaults)
without invoking the callback — the click analogue of ``parse_args``.
"""

from __future__ import annotations

import inspect
import json as _json

import click
import pytest
from click.testing import CliRunner

import vocal_helper as voh
from vocal_helper import cli as cli_mod
from vocal_helper.cli import _pipeline_config, cli

# A resolved engine descriptor, as ``resolve_engine`` (best-engine-ai-helper)
# would return. The CLI threads this into the LLM/EOT stage configs. Tests patch
# ``resolve_engine`` to hand this back so no hardware detection / engine
# resolution runs — the model is never a CLI flag any more.
FAKE_ENGINE = {
    "backend": "ollama",
    "base_url": "http://localhost:11434",
    "llm": {"model": "test-model:latest"},
}

# The shared levers ``_pipeline_config`` consumes. Derived from its signature so
# a newly added shared option flows into these scenarios instead of being
# silently dropped from the parsed params.
_SHARED_LEVERS = tuple(inspect.signature(_pipeline_config).parameters)


def _patch_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the config builder resolve to :data:`FAKE_ENGINE`."""
    monkeypatch.setattr(cli_mod, "resolve_engine", lambda **_: FAKE_ENGINE)


def _parse(command: str, argv: list[str]) -> dict:
    """Parse ``argv`` through the real shipped command without running it.

    Parameters
    ----------
    command : str
        Subcommand name (e.g. ``"mic"``).
    argv : list of str
        Command-line tokens for that subcommand.

    Returns
    -------
    dict
        The parsed parameters, exactly as click would hand them to the callback.
    """
    cmd = cli.commands[command]
    with cmd.make_context(command, list(argv)) as ctx:
        return dict(ctx.params)


def _config(command: str, argv: list[str]):
    """Parse ``argv`` through the real command and build the shipped pipeline config.

    Parameters
    ----------
    command : str
        Subcommand name (e.g. ``"mic"``).
    argv : list of str
        Command-line tokens for that subcommand.

    Returns
    -------
    PipelineConfig
        The config produced by the shipped builder from those options.
    """
    params = _parse(command, argv)
    return _pipeline_config(**{k: v for k, v in params.items() if k in _SHARED_LEVERS})


# ---------------------------------------------------------------------------
# The shipped config builder — vocal_helper.cli._pipeline_config
#
# This is the builder wired into the shipped ``vocal-helper`` entry point, so
# its defaults are what users actually get. Driving it through the real parser
# catches drift between flag defaults and library defaults (auto backend,
# the best-engine-ai-helper-picked analyst, initial_prompt, EOT).
# ---------------------------------------------------------------------------


def test_config_builder_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The builder maps options to ASR / diar blocks and never leaks HF.

    Walks the builder through three coherent facets at once: the bare happy
    path (ASR fields + minimal diar block + no LLM), the optional
    ``--join-threshold`` landing in the diar block, and the guarantee that no
    ``hf_token`` ever appears — weights come from the self-hosted bundle, so an
    ambient ``HF_TOKEN`` must not leak in.
    """
    # Ambient HF_TOKEN must be ignored by the builder (bundle-only weights).
    monkeypatch.setenv("HF_TOKEN", "hf_SHOULD_BE_IGNORED")

    cfg = _config("mic", ["--diar-backend", "pyannote"])
    assert cfg.asr["model"] == "large-v3-turbo-q5_0"
    assert cfg.asr["language"] == "auto"
    assert cfg.asr["threads"] == 6
    assert cfg.diar == {"backend": "pyannote"}  # minimal diar block, no join key
    assert cfg.llm is None  # --llm off → no LLM block at all
    assert "hf_token" not in cfg.diar  # HF is never involved

    # A supplied --join-threshold lands verbatim in the diar block.
    cfg_join = _config("mic", ["--diar-backend", "pyannote", "--join-threshold", "0.42"])
    assert cfg_join.diar["join_threshold"] == 0.42


def test_config_builder_llm_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--llm`` builds the LLM block carrying the resolved engine, not a model tag.

    The model is never a flag: best-engine-ai-helper resolves it from the
    committed brief. The builder threads that engine descriptor into the LLM
    stage config alongside the verbatim window; no ``model`` / ``host`` keys.
    """
    _patch_engine(monkeypatch)

    cfg = _config("mic", ["--llm"])
    assert cfg.llm == {"engine": FAKE_ENGINE, "recent_window_s": 60.0}
    assert "model" not in cfg.llm  # model lives in the engine, not the block
    assert "host" not in cfg.llm


def test_shipped_config_defaults_and_opt_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped builder wires backend/LLM/prompt/EOT defaults and opt-ins correctly.

    One scenario spanning every derived field of the shipped config: the bare
    ``mic`` run defaults the backend to ``auto`` (router-resolved), leaves the
    ASR ``initial_prompt`` empty, and attaches no EOT stage; ``--llm`` carries
    the resolved engine (model comes from the brief, never a flag);
    ``--initial-prompt`` threads into the ASR whisper-bias config; and ``--eot``
    opts into an EOT block carrying the same resolved engine.
    """
    _patch_engine(monkeypatch)

    base = _config("mic", [])
    assert base.diar["backend"] == "auto"  # router decides at run time
    assert base.asr["initial_prompt"] == ""  # generic transcription by default
    assert base.eot is None  # one extra LLM hop is opt-in, not free

    # --llm attaches the analyst with the resolved engine; no model tag flag.
    assert _config("mic", ["--llm"]).llm == {"engine": FAKE_ENGINE, "recent_window_s": 60.0}

    # --initial-prompt reaches the ASR config as the whisper bias prompt.
    prompted = _config("mic", ["--initial-prompt", "telemedicine consult"])
    assert prompted.asr["initial_prompt"] == "telemedicine consult"

    # --eot opts in; the resolved engine is threaded into the SemanticEOTStage.
    assert _config("mic", ["--eot"]).eot == {"engine": FAKE_ENGINE}


# ---------------------------------------------------------------------------
# The real shipped parser — parsing behaviour
# ---------------------------------------------------------------------------


def test_real_parser_parses_and_validates(tmp_path) -> None:
    """The shipped parser defaults ``mic``, round-trips ``file`` flags, and rejects junk.

    Drives the real commands through three parsing facets: ``mic`` with no flags
    (diar backend defaults to ``auto`` — the router decides at run time),
    ``file`` with a positional path plus a mix of value and boolean flags
    round-tripping onto the parsed params, and an unlisted ``--diar-backend``
    value raising a click usage error rather than being silently kept.
    """
    mic = _parse("mic", [])
    # 'auto' means the aiguilleur picks the backend per run (live → nemo).
    assert mic["diar_backend"] == "auto"

    # click.Path(exists=True) guards the positional, so the file must be real.
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    file_params = _parse("file", [str(wav), "--language", "fr", "--llm", "--no-real-time"])
    assert file_params["path"] == str(wav)
    assert file_params["language"] == "fr"
    assert file_params["llm"] is True
    assert file_params["no_real_time"] is True

    # An unknown backend choice must be rejected at parse time, not silently kept.
    with pytest.raises(click.UsageError):
        _parse("mic", ["--diar-backend", "kaldi"])


# ---------------------------------------------------------------------------
# CLI surface — subcommands, help, and the short flags oncall people type
# ---------------------------------------------------------------------------


def test_cli_surface() -> None:
    """The CLI ships the four subcommands and every help screen exits 0.

    Asserts the whole surface in one scenario: the group carries the
    ``{mic, file, url, transcribe}`` subcommands, the group help exits 0 while
    naming the tool, and each subcommand's ``--help`` also exits 0 (no wiring
    bug in any command). The group help must describe what the tool *does* —
    it used to advertise itself as the "click twin of the argparse CLI", an
    architecture that no longer exists.
    """
    expected = {"mic", "file", "url", "transcribe"}
    assert expected.issubset(set(cli.commands.keys()))

    runner = CliRunner()
    group_help = runner.invoke(cli, ["--help"])
    assert group_help.exit_code == 0
    assert "diarized" in group_help.output.lower()
    assert "twin" not in group_help.output.lower()  # no dead architecture in the help

    for sub in sorted(expected):
        result = runner.invoke(cli, [sub, "--help"])
        assert result.exit_code == 0, sub


def test_help_screens_carry_no_rst_markup() -> None:
    """No help screen may show raw ``...`` RST literals to a user.

    click renders a callback's docstring verbatim, so reStructuredText markup
    written for Sphinx reaches the terminal as literal double backticks. Three
    docstrings shipped that way — the group's and ``mic`` / ``url``'s extra
    names — and it is invisible by reading the source, because the markup looks
    correct there. Covers the option help strings too, since they go through the
    same renderer.
    """
    runner = CliRunner()

    screens = {"<group>": runner.invoke(cli, ["--help"]).output}
    for sub in sorted(cli.commands):
        screens[sub] = runner.invoke(cli, [sub, "--help"]).output

    offenders = sorted(name for name, text in screens.items() if "``" in text)
    assert not offenders, f"RST literals leaked into these help screens: {offenders}"


def test_short_help_and_version_flags() -> None:
    """``-h`` mirrors ``--help`` and ``-V`` reports the source version.

    Both short flags are the ones people reach for first and neither is wired by
    click out of the box. ``-V`` is capital because ``-v`` reads as verbose; it
    must report :data:`vocal_helper.__version__` — the version of the code being
    run — rather than the installed distribution metadata, which in an editable
    checkout reports whatever was last pip-installed.
    """
    runner = CliRunner()

    short = runner.invoke(cli, ["-h"])
    long = runner.invoke(cli, ["--help"])
    assert short.exit_code == 0
    assert short.output == long.output

    for flag in ("-V", "--version"):
        result = runner.invoke(cli, [flag])
        assert result.exit_code == 0, flag
        assert "vocal-helper" in result.output
        assert voh.__version__ in result.output, flag


# ---------------------------------------------------------------------------
# The router choke-point the CLI and the HTTP surface share.
# ---------------------------------------------------------------------------


def test_router_backend_resolution() -> None:
    """The router resolves ``auto`` for a live stream and honours explicit backends.

    Exercises both branches of ``_route_backend``: a live stream with ``auto``
    resolves to ``nemo`` (the shipped online default — best online embedder at
    every length) with a note that surfaces the speed axis, while an explicit
    backend passes through verbatim with no note (an operator override must not
    be second-guessed).
    """
    from vocal_helper.cli import _route_backend

    backend, note = _route_backend(requested_backend="auto", live=True, duration_s=None)
    assert backend == "nemo"
    assert note and "RTF" in note  # speed axis surfaced alongside quality

    backend2, note2 = _route_backend(requested_backend="pyannote", live=True, duration_s=None)
    assert backend2 == "pyannote"
    assert note2 is None  # explicit choice → no router note


# ---------------------------------------------------------------------------
# Argument validation, error exit codes, and a mocked happy path.
#
# These drive the *shipped* commands through click's CliRunner so the whole
# option stack (Path(exists=True), Choice, version) is exercised. Every model /
# decode boundary is mocked, so nothing here touches ffmpeg, numpy of real
# size, whisper.cpp, or a diarizer — only the CLI plumbing is under test.
# ---------------------------------------------------------------------------


def test_rejects_missing_path() -> None:
    """``file`` / ``transcribe`` with a non-existent path must exit non-zero (usage error).

    ``click.Path(exists=True)`` is the guard: a path the user typo'd or that was
    deleted should be caught by argument validation *before* any source opens or
    any model loads. click reports this as exit code 2 and mentions the bad path,
    so the user gets an actionable message instead of a deep stack trace.
    """
    runner = CliRunner()

    # Both file-consuming commands share the same existence guard; check each.
    for sub in ("file", "transcribe"):
        result = runner.invoke(cli, [sub, "/no/such/file.wav"])
        # Exit 2 is click's canonical "usage error" code.
        assert result.exit_code == 2, sub
        # The offending path is surfaced so the message is actionable.
        assert "/no/such/file.wav" in result.output


def test_rejects_unknown_diar_backend() -> None:
    """An out-of-choice ``--diar-backend`` must be rejected, not silently kept.

    ``--diar-backend`` is a ``click.Choice`` over the supported engines. A value
    outside that set (a stale ``kaldi`` from muscle memory) has to fail at parse
    time with exit 2 — otherwise the bad backend would only blow up later when a
    stage tries to load it.
    """
    runner = CliRunner()

    result = runner.invoke(cli, ["mic", "--diar-backend", "kaldi"])
    assert result.exit_code == 2
    # click echoes the invalid value in its Choice error.
    assert "kaldi" in result.output


def test_main_prints_clean_error_on_a_real_library_exception(
    capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The ``main()`` console-script wrapper turns a library exception into exit 1.

    ``CliRunner.invoke`` (used by every other test here) catches exceptions
    itself, so it never exercises ``main()`` — the actual ``vocal-helper`` entry
    point. Drive it directly: a real WAV path passes ``click.Path(exists=True)``,
    then a mocked ``load_audio`` raises, which used to propagate as a raw
    traceback instead of a clean ``Error: ...`` + exit 1.
    """
    import audio_helper

    from vocal_helper.cli import main

    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"not a real wav")

    def _boom(*_a, **_kw):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(audio_helper, "load_audio", _boom)
    monkeypatch.setattr("sys.argv", ["vocal-helper", "transcribe", str(wav)])

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Error:" in capsys.readouterr().err


def test_transcribe_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """``transcribe`` decodes + transcribes a real path, printing plain text or JSONL.

    The path must exist (``Path(exists=True)``) so we create an empty file, but
    the decode (``audio_helper.load_audio``) and the ASR call
    (``transcribe_pcm``) are both mocked — no ffmpeg, no whisper.cpp. This pins
    the command's two output modes: bare text on stdout by default, and a
    ``{"path", "text"}`` JSON object under ``--jsonl``.
    """
    import audio_helper
    import numpy as np

    from vocal_helper import asr as asr_mod

    # A real (empty) file so click's existence check passes; content is unused
    # because the decoder is stubbed out below.
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"")

    # Decode → a trivial 1-sample buffer at 16 kHz; never invokes ffmpeg.
    monkeypatch.setattr(
        audio_helper, "load_audio", lambda *a, **k: (np.zeros(1, dtype=np.float32), 16_000)
    )
    # ASR → a fixed transcript; never loads whisper.cpp weights.
    monkeypatch.setattr(asr_mod, "transcribe_pcm", lambda **k: "hello world")

    runner = CliRunner()

    # Default mode → plain text on stdout.
    plain = runner.invoke(cli, ["transcribe", str(wav)])
    assert plain.exit_code == 0
    assert plain.output.strip() == "hello world"

    # --jsonl mode → a single JSON object carrying the path and the transcript.
    jsonl = runner.invoke(cli, ["transcribe", str(wav), "--jsonl"])
    assert jsonl.exit_code == 0
    payload = _json.loads(jsonl.output.strip())
    assert payload == {"path": str(wav), "text": "hello world"}


def test_print_event_renders_both_event_shapes(capsys: pytest.CaptureFixture) -> None:
    """``_print_event`` renders utterance and summary events in both output modes.

    The printer is the CLI's rendering choke-point, so it deserves a direct test.
    An utterance event (``text`` key) becomes a timestamped speaker line; a
    summary event (``summary`` key) becomes the rolling-digest block. In ``jsonl``
    mode both collapse to a single JSON line with the raw ``pcm`` stripped (it is
    un-serialisable and pointless in a log).
    """
    from vocal_helper.cli import _print_event

    utterance = {"t0": 0.0, "t1": 1.5, "speaker": "S1", "text": "hi", "pcm": [0.0]}
    summary = {"t0": 12.0, "model": "gemma3:4b", "summary": "so far", "recent": "verbatim"}

    # Human-readable utterance line carries the timestamps, speaker, and text.
    _print_event(utterance, jsonl=False)
    out = capsys.readouterr().out
    assert "S1" in out and "hi" in out and "0.00s" in out

    # Human-readable summary block carries the model tag and both text panels.
    _print_event(summary, jsonl=False)
    out = capsys.readouterr().out
    assert "rolling summary" in out and "gemma3:4b" in out and "verbatim" in out

    # JSONL mode emits one line and drops the un-serialisable raw PCM.
    _print_event(utterance, jsonl=True)
    line = _json.loads(capsys.readouterr().out.strip())
    assert "pcm" not in line and line["text"] == "hi"
