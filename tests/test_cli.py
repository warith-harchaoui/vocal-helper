"""Tests for the ``vocal-helper`` CLI surface.

Construction-only — we never call ``asyncio.run(_amain(...))`` because
that would require a real source and load the heavy stages. The goal
is to assert that argparse, the config builder and the HF-token
resolution path do the right thing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from vocal_helper import cli


def _ns(**overrides: object) -> argparse.Namespace:
    """Build a minimal Namespace matching ``add_common`` defaults."""
    base: dict[str, object] = {
        "whisper_model": "large-v3-turbo-q5_0",
        "language": "auto",
        "threads": 6,
        "diar_backend": "pyannote",
        "hf_token": None,
        "join_threshold": None,
        "llm": False,
        "llm_model": "gemma4:e4b",
        "llm_recent_window_s": 60.0,
        "ollama_host": None,
        "jsonl": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# _build_config — happy paths
# ---------------------------------------------------------------------------


def test_build_config_minimal() -> None:
    cfg = cli._build_config(_ns())
    assert cfg.asr["model"] == "large-v3-turbo-q5_0"
    assert cfg.asr["language"] == "auto"
    assert cfg.asr["threads"] == 6
    assert cfg.diar == {"backend": "pyannote"}
    assert cfg.llm is None


def test_build_config_threads_through_join_threshold() -> None:
    cfg = cli._build_config(_ns(join_threshold=0.42))
    assert cfg.diar["join_threshold"] == 0.42


def test_build_config_llm_block_only_when_enabled() -> None:
    cfg = cli._build_config(_ns(llm=True, ollama_host="http://localhost:11434"))
    assert cfg.llm == {
        "model": "gemma4:e4b",
        "recent_window_s": 60.0,
        "host": "http://localhost:11434",
    }


def test_build_config_llm_block_without_host() -> None:
    cfg = cli._build_config(_ns(llm=True))
    assert cfg.llm == {"model": "gemma4:e4b", "recent_window_s": 60.0}
    assert "host" not in cfg.llm


# ---------------------------------------------------------------------------
# _build_config — HF token resolution order
# ---------------------------------------------------------------------------


def test_build_config_explicit_flag_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_FROM_ENV")
    cfg = cli._build_config(_ns(hf_token="hf_FROM_CLI"))
    assert cfg.diar["hf_token"] == "hf_FROM_CLI"


def test_build_config_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_FROM_ENV")
    cfg = cli._build_config(_ns())
    assert cfg.diar["hf_token"] == "hf_FROM_ENV"


def test_build_config_falls_back_to_settings_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text("secrets:\n  hf_token: hf_FROM_FILE\n", encoding="utf-8")
    monkeypatch.setenv("VOCAL_HELPER_SETTINGS", str(settings))
    cfg = cli._build_config(_ns())
    assert cfg.diar["hf_token"] == "hf_FROM_FILE"


def test_build_config_omits_hf_token_when_unset() -> None:
    """No source provides a token → the diar dict has no ``hf_token`` key
    so :class:`OnlineDiarStage` keeps its own default (``None``)."""
    cfg = cli._build_config(_ns())
    assert "hf_token" not in cfg.diar


# ---------------------------------------------------------------------------
# argparse smoke — make sure the parser actually parses what we test above.
# Reaches into the ``main`` function only enough to grab the parser.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Re-build the CLI parser the same way ``cli.main`` does."""
    p = argparse.ArgumentParser(prog="vocal-helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--whisper-model", default="large-v3-turbo-q5_0")
        sp.add_argument("--language", default="auto")
        sp.add_argument("--threads", type=int, default=6)
        sp.add_argument("--diar-backend", choices=["pyannote", "nemo"], default="pyannote")
        sp.add_argument("--hf-token", default=None)
        sp.add_argument("--join-threshold", type=float, default=None)
        sp.add_argument("--llm", action="store_true")
        sp.add_argument("--llm-model", default="gemma4:e4b")
        sp.add_argument("--llm-recent-window-s", type=float, default=60.0)
        sp.add_argument("--ollama-host", default=None)
        sp.add_argument("--jsonl", action="store_true")

    mic = sub.add_parser("mic")
    add_common(mic)
    mic.add_argument("--device", default=None)

    f = sub.add_parser("file")
    add_common(f)
    f.add_argument("path", type=str)
    f.add_argument("--no-real-time", action="store_true")
    f.add_argument("--offline", action="store_true")
    return p


def test_cli_parser_mic_minimal() -> None:
    args = _build_parser().parse_args(["mic"])
    assert args.cmd == "mic"
    assert args.diar_backend == "pyannote"
    assert args.hf_token is None


def test_cli_parser_file_with_overrides() -> None:
    args = _build_parser().parse_args([
        "file", "/tmp/in.wav",
        "--language", "fr",
        "--hf-token", "hf_xyz",
        "--llm",
        "--no-real-time",
    ])
    assert args.cmd == "file"
    assert args.path == "/tmp/in.wav"
    assert args.language == "fr"
    assert args.hf_token == "hf_xyz"
    assert args.llm is True
    assert args.no_real_time is True


def test_cli_parser_rejects_unknown_backend() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["mic", "--diar-backend", "kaldi"])


# ---------------------------------------------------------------------------
# argparse surface tests — vocal_helper.cli_argparse
#
# These assert that the canonical argparse CLI ships the four expected
# subcommands (mic / file / url / transcribe) and that each subcommand's
# ``--help`` exits cleanly. No pipeline is actually started.
# ---------------------------------------------------------------------------


def test_argparse_parser_builds_without_error() -> None:
    """Building the parser should never fail (imports, subcommand wiring)."""
    from vocal_helper.cli_argparse import build_parser

    parser = build_parser()
    subparsers_action = next(
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    )
    expected = {"mic", "file", "url", "transcribe"}
    assert expected.issubset(set(subparsers_action.choices.keys()))


def test_argparse_help_exits_zero(capsys: pytest.CaptureFixture) -> None:
    """``vocal-helper --help`` should exit with code 0 and print usage."""
    from vocal_helper.cli_argparse import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "vocal-helper" in captured.out.lower()


@pytest.mark.parametrize("sub", ["mic", "file", "url", "transcribe"])
def test_argparse_subcommand_help_exits_zero(sub: str) -> None:
    """Every subcommand's ``--help`` should exit 0 (no wiring bug)."""
    from vocal_helper.cli_argparse import main

    with pytest.raises(SystemExit) as exc:
        main([sub, "--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Click surface tests — vocal_helper.cli_click
# ---------------------------------------------------------------------------


def test_click_group_has_expected_subcommands() -> None:
    """The click group must expose the same subcommands as the argparse CLI."""
    # ``click`` lives in the optional [cli] extra. Skip cleanly if absent.
    _click = pytest.importorskip("click")

    from vocal_helper.cli_click import cli as click_cli

    expected = {"mic", "file", "url", "transcribe"}
    assert expected.issubset(set(click_cli.commands.keys()))


def test_click_help_exits_zero() -> None:
    """``vocal-helper-click --help`` should exit 0."""
    _click = pytest.importorskip("click")
    from click.testing import CliRunner

    from vocal_helper.cli_click import cli as click_cli

    runner = CliRunner()
    result = runner.invoke(click_cli, ["--help"])
    assert result.exit_code == 0
    assert "vocal helper" in result.output.lower()


@pytest.mark.parametrize("sub", ["mic", "file", "url", "transcribe"])
def test_click_subcommand_help_exits_zero(sub: str) -> None:
    """Every click subcommand's ``--help`` should exit 0."""
    _click = pytest.importorskip("click")
    from click.testing import CliRunner

    from vocal_helper.cli_click import cli as click_cli

    runner = CliRunner()
    result = runner.invoke(click_cli, [sub, "--help"])
    assert result.exit_code == 0
