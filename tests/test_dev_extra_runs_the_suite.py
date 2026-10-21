"""
The `[dev]` extra must be able to run this test suite.

CI installs `pip install -e '.[dev]'` and nothing else, so a dependency the
tests need and the extra does not name is not a skipped test — it is a red
build. Two of them sat in this repository at once:

- the fan-out and strict-pipeline tests are `async def` carrying
  `@pytest.mark.asyncio`, and pytest **fails** an async test it has no plugin
  for rather than skipping it: eleven red tests for one missing line;
- `tests/test_bench.py` asserts real WER figures, and `jiwer` was declared
  nowhere in `pyproject.toml` at all — not even as an extra — while
  `bench.wer_report`, a public function, imported it bare.

Both were invisible on a developer machine, where the packages happened to be
installed. These tests tie the declaration to the need, so adding an async
test or a metric-backed assertion without the dependency fails here — on the
machine of whoever added it — instead of in CI.

Usage Example
-------------
>>> #   pytest tests/test_dev_extra_runs_the_suite.py

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TESTS = Path(__file__).resolve().parent


def _dev_extra() -> str:
    """The text of the `dev = [...]` extra, read without tomllib (3.10 floor)."""
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dev = \[(.*?)^\]", text, re.S | re.M)
    assert block, "pyproject.toml has no `dev = [...]` extra"
    return block.group(1)


def test_async_tests_come_with_the_plugin_that_runs_them() -> None:
    users = sorted(
        path.name
        for path in _TESTS.glob("test_*.py")
        if "pytest.mark.asyncio" in path.read_text(encoding="utf-8")
    )
    if not users:
        return  # nothing async in the suite; the plugin is not owed
    assert "pytest-asyncio" in _dev_extra(), (
        f"{users} carry @pytest.mark.asyncio, which pytest FAILS without the plugin; "
        "add pytest-asyncio to the [dev] extra"
    )


def test_the_word_error_metrics_come_with_jiwer() -> None:
    """`bench` is only importable; its metrics need jiwer to compute anything."""
    users = sorted(
        path.name
        for path in _TESTS.glob("test_*.py")
        if re.search(r"\b(wer_report|cpwer_report)\s*\(", path.read_text(encoding="utf-8"))
    )
    if not users:
        return
    assert "jiwer" in _dev_extra(), (
        f"{users} call the word-error metrics; add jiwer to the [dev] extra"
    )


def test_the_metrics_name_their_extra_rather_than_failing_bare() -> None:
    """A user who installed without `[bench]` must be told which extra to add.

    A naked ``ModuleNotFoundError: No module named 'jiwer'`` out of a public
    function tells the caller nothing about what to install.
    """
    source = (_ROOT / "vocal_helper" / "bench.py").read_text(encoding="utf-8")
    assert "vocal-helper[bench]" in source, (
        "bench.py imports jiwer without naming the extra that provides it"
    )
    bare = re.findall(r"^\s{4}import jiwer$", source, re.M)
    assert not bare, "a bare `import jiwer` is back; route it through _jiwer()"
