"""Tests for the ``vocal-helper`` CLI surface.

Construction-only — we never call ``asyncio.run(_amain(...))`` because
that would require a real source and load the heavy stages. The goal
is to assert that argparse and the config builders do the right thing
(no HuggingFace token is involved anywhere).

The tests are organised as a handful of scenarios that each drive a
realistic slice of the CLI end-to-end — parse representative ``argv``
through the *real* shipped parser, build the pipeline config, and assert
the derived fields together — rather than one micro-test per flag.
"""

from __future__ import annotations

import argparse

import pytest

from vocal_helper import cli


def _ns(**overrides: object) -> argparse.Namespace:
    """Build a minimal Namespace matching ``add_common`` defaults.

    Parameters
    ----------
    **overrides
        Fields to override on top of the ``add_common`` defaults.

    Returns
    -------
    argparse.Namespace
        A namespace ready to feed :func:`vocal_helper.cli._build_config`.
    """
    base: dict[str, object] = {
        "whisper_model": "large-v3-turbo-q5_0",
        "language": "auto",
        "threads": 6,
        "diar_backend": "pyannote",
        "join_threshold": None,
        "llm": False,
        "llm_model": "gemma4:e4b",
        "llm_recent_window_s": 60.0,
        "ollama_host": None,
        "jsonl": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _real_parser() -> argparse.ArgumentParser:
    """Return the real shipped argparse parser so tests never drift from it.

    Returns
    -------
    argparse.ArgumentParser
        The parser wired into the ``vocal-helper`` entry point.
    """
    from vocal_helper.cli_argparse import build_parser

    return build_parser()


def _argparse_config(argv: list[str]):
    """Parse ``argv`` through the real parser and build the shipped pipeline config.

    Parameters
    ----------
    argv
        Command-line tokens (e.g. ``["mic", "--llm"]``).

    Returns
    -------
    object
        The pipeline config produced by the shipped builder.
    """
    from vocal_helper.cli_argparse import _build_pipeline_config, build_parser

    args = build_parser().parse_args(argv)
    return _build_pipeline_config(args)


# ---------------------------------------------------------------------------
# cli._build_config — the legacy builder (ASR / diar / llm blocks + no HF token)
# ---------------------------------------------------------------------------


def test_build_config_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cli._build_config`` maps a namespace to ASR / diar blocks and never leaks HF.

    Walks the legacy builder through three coherent facets at once: the bare
    happy path (ASR fields + minimal diar block + no LLM), the optional
    ``--join-threshold`` landing in the diar block, and the guarantee that no
    ``hf_token`` ever appears — weights come from the self-hosted bundle, so an
    ambient ``HF_TOKEN`` must not leak in.
    """
    # Ambient HF_TOKEN must be ignored by the builder (bundle-only weights).
    monkeypatch.setenv("HF_TOKEN", "hf_SHOULD_BE_IGNORED")

    cfg = cli._build_config(_ns())
    assert cfg.asr["model"] == "large-v3-turbo-q5_0"
    assert cfg.asr["language"] == "auto"
    assert cfg.asr["threads"] == 6
    assert cfg.diar == {"backend": "pyannote"}  # minimal diar block, no join key
    assert cfg.llm is None  # --llm off → no LLM block at all
    assert "hf_token" not in cfg.diar  # HF is never involved

    # A supplied --join-threshold lands verbatim in the diar block.
    cfg_join = cli._build_config(_ns(join_threshold=0.42))
    assert cfg_join.diar["join_threshold"] == 0.42


def test_build_config_llm_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--llm`` builds the LLM block; ``host`` is present only when supplied.

    Covers both LLM shapes in one scenario: with ``--ollama-host`` the block
    carries the full three keys, and without it the ``host`` key is *omitted*
    (not set to ``None``), so downstream Ollama defaults apply.
    """
    cfg_host = cli._build_config(_ns(llm=True, ollama_host="http://localhost:11434"))
    assert cfg_host.llm == {
        "model": "gemma4:e4b",
        "recent_window_s": 60.0,
        "host": "http://localhost:11434",
    }

    cfg_nohost = cli._build_config(_ns(llm=True))
    assert cfg_nohost.llm == {"model": "gemma4:e4b", "recent_window_s": 60.0}
    assert "host" not in cfg_nohost.llm  # omitted, not None


# ---------------------------------------------------------------------------
# The real shipped argparse parser — parsing behaviour
# ---------------------------------------------------------------------------


def test_real_parser_parses_and_validates() -> None:
    """The shipped parser defaults ``mic``, round-trips ``file`` flags, and rejects junk.

    Drives the real ``build_parser`` through three parsing facets: ``mic`` with
    no flags (diar backend defaults to ``auto`` — the router decides at run
    time), ``file`` with a positional path plus a mix of value and boolean
    flags round-tripping onto the namespace, and an unlisted ``--diar-backend``
    value making argparse ``SystemExit``.
    """
    mic = _real_parser().parse_args(["mic"])
    assert mic.command == "mic"
    # 'auto' means the aiguilleur picks the backend per run (live → nemo).
    assert mic.diar_backend == "auto"

    file_args = _real_parser().parse_args(
        ["file", "/tmp/in.wav", "--language", "fr", "--llm", "--no-real-time"]
    )
    assert file_args.command == "file"
    assert file_args.path == "/tmp/in.wav"
    assert file_args.language == "fr"
    assert file_args.llm is True
    assert file_args.no_real_time is True

    # An unknown backend choice must be rejected by argparse, not silently kept.
    with pytest.raises(SystemExit):
        _real_parser().parse_args(["mic", "--diar-backend", "kaldi"])


# ---------------------------------------------------------------------------
# CLI surface — both frontends expose the same subcommands and clean --help
# ---------------------------------------------------------------------------


def test_argparse_cli_surface(capsys: pytest.CaptureFixture) -> None:
    """The argparse CLI ships the four subcommands and every ``--help`` exits 0.

    Asserts the whole argparse surface in one scenario: the parser builds with
    the ``{mic, file, url, transcribe}`` subcommands wired in, top-level
    ``--help`` exits 0 while printing usage, and each subcommand's ``--help``
    also exits 0 (no wiring bug in any subparser).
    """
    from vocal_helper.cli_argparse import build_parser, main

    parser = build_parser()
    subparsers_action = next(
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    )
    expected = {"mic", "file", "url", "transcribe"}
    assert expected.issubset(set(subparsers_action.choices.keys()))

    # Top-level --help exits cleanly and prints the program name.
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "vocal-helper" in capsys.readouterr().out.lower()

    # Every subcommand's --help exits 0 — proves each subparser is wired.
    for sub in sorted(expected):
        with pytest.raises(SystemExit) as sub_exc:
            main([sub, "--help"])
        assert sub_exc.value.code == 0, sub


def test_click_cli_surface() -> None:
    """The optional click CLI mirrors the argparse subcommands with clean ``--help``.

    ``click`` lives in the ``[cli]`` extra, so the whole scenario is skipped when
    absent. When present it must expose the same ``{mic, file, url, transcribe}``
    subcommands and exit 0 for both the group ``--help`` and every subcommand
    ``--help`` — i.e. stay in lockstep with the argparse frontend.
    """
    pytest.importorskip("click")  # optional [cli] extra — skip cleanly if absent
    from click.testing import CliRunner

    from vocal_helper.cli_click import cli as click_cli

    expected = {"mic", "file", "url", "transcribe"}
    assert expected.issubset(set(click_cli.commands.keys()))

    runner = CliRunner()
    group_help = runner.invoke(click_cli, ["--help"])
    assert group_help.exit_code == 0
    assert "vocal helper" in group_help.output.lower()

    for sub in sorted(expected):
        result = runner.invoke(click_cli, [sub, "--help"])
        assert result.exit_code == 0, sub


# ---------------------------------------------------------------------------
# Canonical shipped config builder — vocal_helper.cli_argparse._build_pipeline_config
#
# This is the builder wired into the shipped ``vocal-helper`` entry point, so
# its defaults are what users actually get. Driving it through the real parser
# catches drift between flag defaults and library defaults (auto backend,
# gemma3:4b analyst, initial_prompt, EOT).
# ---------------------------------------------------------------------------


def test_argparse_file_config_end_to_end() -> None:
    """The shipped builder wires backend/LLM/prompt/EOT defaults and opt-ins correctly.

    One scenario spanning every derived field of the shipped config: the bare
    ``mic`` run defaults the backend to ``auto`` (router-resolved), leaves the
    ASR ``initial_prompt`` empty, and attaches no EOT stage; ``--llm`` defaults
    to the Pareto-best ``gemma3:4b`` analyst; ``--initial-prompt`` threads into
    the ASR whisper-bias config; and ``--eot`` opts into an EOT block keyed by
    ``eot_model``.
    """
    base = _argparse_config(["mic"])
    assert base.diar["backend"] == "auto"  # router decides at run time
    assert base.asr["initial_prompt"] == ""  # generic transcription by default
    assert base.eot is None  # one extra LLM hop is opt-in, not free

    # --llm defaults to the gemma3:4b analyst with a 60 s recency window.
    assert _argparse_config(["mic", "--llm"]).llm == {
        "model": "gemma3:4b",
        "recent_window_s": 60.0,
    }

    # --initial-prompt reaches the ASR config as the whisper bias prompt.
    prompted = _argparse_config(["mic", "--initial-prompt", "telemedicine consult"])
    assert prompted.asr["initial_prompt"] == "telemedicine consult"

    # --eot opts in; --eot-model is stored under the SemanticEOTStage key.
    eot = _argparse_config(["mic", "--eot", "--eot-model", "qwen2.5:3b"])
    assert eot.eot == {"eot_model": "qwen2.5:3b"}


def test_router_backend_resolution() -> None:
    """The router resolves ``auto`` for a live stream and honours explicit backends.

    Exercises both branches of ``_route_backend``: a live stream with ``auto``
    resolves to ``nemo`` (the shipped online default — best online embedder at
    every length) with a note that surfaces the speed axis, while an explicit
    backend passes through verbatim with no note (an operator override must not
    be second-guessed).
    """
    from vocal_helper.cli_argparse import _route_backend

    backend, note = _route_backend(requested_backend="auto", live=True, duration_s=None)
    assert backend == "nemo"
    assert note and "RTF" in note  # speed axis surfaced alongside quality

    backend2, note2 = _route_backend(requested_backend="pyannote", live=True, duration_s=None)
    assert backend2 == "pyannote"
    assert note2 is None  # explicit choice → no router note


# ---------------------------------------------------------------------------
# click CLI — argument validation, error exit codes, and a mocked happy path.
#
# These drive the *shipped* click commands through click's CliRunner so the
# whole option stack (Path(exists=True), Choice, version) is exercised. Every
# model / decode boundary is mocked, so nothing here touches ffmpeg, numpy of
# real size, whisper.cpp, or a diarizer — only the CLI plumbing is under test.
# ---------------------------------------------------------------------------


def _click_runner():
    """Return (click_cli, CliRunner) or skip cleanly when the [cli] extra is absent."""
    pytest.importorskip("click")  # optional [cli] extra
    from click.testing import CliRunner

    from vocal_helper.cli_click import cli as click_cli

    return click_cli, CliRunner()


def test_click_rejects_missing_path() -> None:
    """``file`` / ``transcribe`` with a non-existent path must exit non-zero (usage error).

    ``click.Path(exists=True)`` is the guard: a path the user typo'd or that was
    deleted should be caught by argument validation *before* any source opens or
    any model loads. click reports this as exit code 2 and mentions the bad path,
    so the user gets an actionable message instead of a deep stack trace.
    """
    click_cli, runner = _click_runner()

    # Both file-consuming commands share the same existence guard; check each.
    for sub in ("file", "transcribe"):
        result = runner.invoke(click_cli, [sub, "/no/such/file.wav"])
        # Exit 2 is click's canonical "usage error" code.
        assert result.exit_code == 2, sub
        # The offending path is surfaced so the message is actionable.
        assert "/no/such/file.wav" in result.output


def test_click_rejects_unknown_diar_backend() -> None:
    """An out-of-choice ``--diar-backend`` must be rejected by click, not silently kept.

    ``--diar-backend`` is a ``click.Choice`` over the supported engines. A value
    outside that set (a stale ``kaldi`` from muscle memory) has to fail at parse
    time with exit 2, mirroring the argparse twin's rejection — otherwise the bad
    backend would only blow up later when a stage tries to load it.
    """
    click_cli, runner = _click_runner()

    result = runner.invoke(click_cli, ["mic", "--diar-backend", "kaldi"])
    assert result.exit_code == 2
    # click echoes the invalid value in its Choice error.
    assert "kaldi" in result.output


def test_click_version_flag_exits_zero() -> None:
    """``--version`` prints the program name + version and exits cleanly.

    ``version_option`` is wired at the group; this proves the package metadata
    lookup resolves (a broken ``package_name`` would raise here) and that the
    prog name matches the ``vocal-helper-click`` entry point.
    """
    click_cli, runner = _click_runner()

    result = runner.invoke(click_cli, ["--version"])
    assert result.exit_code == 0
    assert "vocal-helper-click" in result.output


def test_click_transcribe_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """``transcribe`` decodes + transcribes a real path, printing plain text or JSONL.

    The path must exist (``Path(exists=True)``) so we create an empty file, but
    the decode (``audio_helper.load_audio``) and the ASR call
    (``transcribe_pcm``) are both mocked — no ffmpeg, no whisper.cpp. This pins
    the command's two output modes: bare text on stdout by default, and a
    ``{"path", "text"}`` JSON object under ``--jsonl``.
    """
    pytest.importorskip("click")
    import json as _json

    import audio_helper
    import numpy as np
    from click.testing import CliRunner

    from vocal_helper import asr as asr_mod
    from vocal_helper.cli_click import cli as click_cli

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
    plain = runner.invoke(click_cli, ["transcribe", str(wav)])
    assert plain.exit_code == 0
    assert plain.output.strip() == "hello world"

    # --jsonl mode → a single JSON object carrying the path and the transcript.
    jsonl = runner.invoke(click_cli, ["transcribe", str(wav), "--jsonl"])
    assert jsonl.exit_code == 0
    payload = _json.loads(jsonl.output.strip())
    assert payload == {"path": str(wav), "text": "hello world"}


def test_click_print_event_renders_both_event_shapes(capsys: pytest.CaptureFixture) -> None:
    """``_print_event`` renders utterance and summary events in both output modes.

    The printer is the CLI's rendering choke-point, so it deserves a direct test.
    An utterance event (``text`` key) becomes a timestamped speaker line; a
    summary event (``summary`` key) becomes the rolling-digest block. In ``jsonl``
    mode both collapse to a single JSON line with the raw ``pcm`` stripped (it is
    un-serialisable and pointless in a log).
    """
    pytest.importorskip("click")
    from vocal_helper.cli_click import _print_event

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
    import json as _json

    line = _json.loads(capsys.readouterr().out.strip())
    assert "pcm" not in line and line["text"] == "hi"
