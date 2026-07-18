"""
Unit tests for the supported-language (``lang_pair``) resolver.

Module summary
--------------
Exercises :func:`vocal_helper.lid.resolve_lang_pair` and
:func:`vocal_helper.lid.languages_from_i18n` — the helpers that enforce the
project's FR+EN language floor while letting extra languages be added freely. This
is about a caller's *supported / output* languages (the ``lang_pair`` convention
shared with the front-audio skill and notes-helper's ``locales/i18n.yaml``), not
about restricting audio language detection, which stays a-priori-free. Model-free.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

from pathlib import Path

from vocal_helper.lid import (
    DEFAULT_LANG_PAIR,
    languages_from_i18n,
    resolve_lang_pair,
)


def test_default_is_fr_en_floor() -> None:
    """No spec yields exactly the FR+EN floor, French first."""
    assert resolve_lang_pair() == ("fr", "en")
    assert DEFAULT_LANG_PAIR == ("fr", "en")


def test_string_spec_is_normalised_and_floored() -> None:
    """Comma/space specs parse, lower-case, and keep the floor first."""
    assert resolve_lang_pair("en,fr") == ("fr", "en")
    assert resolve_lang_pair("EN FR") == ("fr", "en")


def test_extra_language_is_added_not_blocked() -> None:
    """Adding a language never removes the floor — it appends after it."""
    assert resolve_lang_pair("es") == ("fr", "en", "es")
    assert resolve_lang_pair(["de", "en"]) == ("fr", "en", "de")


def test_duplicates_collapse() -> None:
    """Repeated codes appear once, order preserved."""
    assert resolve_lang_pair("fr,fr,es,es") == ("fr", "en", "es")


def test_languages_from_i18n_reads_meta(tmp_path: Path) -> None:
    """``meta.languages`` keys drive the supported set, with the floor guaranteed.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory holding a small fake catalog.
    """
    cat = tmp_path / "i18n.yaml"
    cat.write_text(
        "meta:\n  languages:\n    en: English\n    fr: Français\n    de: Deutsch\n",
        encoding="utf-8",
    )
    assert languages_from_i18n(str(cat)) == ("fr", "en", "de")


def test_languages_from_i18n_missing_file_falls_back(tmp_path: Path) -> None:
    """A missing catalog degrades to the FR+EN floor, never an error."""
    assert languages_from_i18n(str(tmp_path / "nope.yaml")) == ("fr", "en")
