"""The pyannote offline checkpoint knob — which weights, and why that default.

No model is loaded here: the construction-time contract is what these lock.
"""

import pytest

from vocal_helper.diar import _PyannoteOfflineDiar as Pya


def test_the_default_is_community_1_because_it_has_no_catastrophic_case() -> None:
    """community-1 and 3.1 tie on median DER; the tail is what separates them.

    On the 2026-10-19 AMI bench both land at 0.116 median, so a median-only comparison
    says there is nothing to choose. 3.1's worst meeting is 0.474 and community-1's is
    0.149 — same centre, a spread four times shorter.
    """
    assert Pya.DEFAULT_CHECKPOINT == "community-1"
    assert Pya().checkpoint == "community-1"


def test_the_legacy_checkpoint_is_still_selectable() -> None:
    assert Pya(checkpoint="3.1").checkpoint == "3.1"


def test_an_unknown_checkpoint_is_refused_at_construction() -> None:
    """Fail where the mistake was made, not later inside a model load."""
    with pytest.raises(ValueError, match="unknown pyannote checkpoint"):
        Pya(checkpoint="4.2")


def test_checkpoint_names_are_case_and_space_insensitive() -> None:
    assert Pya(checkpoint="  Community-1 ").checkpoint == "community-1"


def test_every_checkpoint_maps_to_a_bundle_folder_and_a_config() -> None:
    """Both are self-contained folders whose config points at its own weights."""
    for name, (sub, cfg) in Pya._CHECKPOINTS.items():
        assert sub and cfg.endswith(".yaml"), name
    assert Pya.DEFAULT_CHECKPOINT in Pya._CHECKPOINTS
