"""
Embedding audio longer than the sherpa ONNX export can swallow in one call.

Module summary
--------------
The mirrored TitaNet-large ONNX graph carries a **static** attention-mask length
of 12288 frames — 122.88 s at its 10 ms hop — and one frame past it the graph
raises instead of degrading ("Attempting to broadcast an axis by a dimension
other than 1. 12288 by 12298"). It is a property of the export, so it cannot be
configured away, and the callers that hit it were not told: one swallowed the
exception and dropped the speaker, which DER then booked as missed speech, and
the likeliest victim was whoever talked the most. These tests pin the windowing
that replaced that silence. No model is loaded: ``_embed_one`` is stubbed, so
what is under test is the arithmetic, not onnxruntime.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import numpy as np
import pytest

from vocal_helper.diar import SHERPA_MAX_EMBED_S, _SherpaEmbedder

SR = 16_000


def _embedder(returns=None):
    """A sherpa embedder whose one-shot call is stubbed, plus the log of its inputs."""
    emb = _SherpaEmbedder(model_path="/nonexistent.onnx")
    seen: list[int] = []
    queue = list(returns or [])

    def fake_embed_one(pcm, sr):
        seen.append(int(pcm.shape[0]))
        if queue:
            return np.asarray(queue.pop(0), dtype=np.float32)
        return np.full(192, 0.1, dtype=np.float32)

    emb._embed_one = fake_embed_one  # type: ignore[method-assign]
    return emb, seen


def _noise(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_the_cap_stays_under_the_static_mask_of_the_export() -> None:
    """122.88 s = 12288 frames x 10 ms hop. Measured: 120 s embeds, 123 s raises.

    Pinned because the margin is the whole safety of the thing: a cap set *at*
    the mask length would raise on frame-count rounding alone.
    """
    assert SHERPA_MAX_EMBED_S < 122.88


def test_below_the_cap_nothing_changed() -> None:
    """One call, result passed through untouched — the windowing must not exist here.

    The stricter half of the contract. A voiceprint cache is clustered ACROSS
    episodes, so a pooled vector and a whole-buffer vector are not the same
    statistic; silently pooling short audio too would move every number already
    measured with this embedder.
    """
    want = np.linspace(-1.0, 1.0, 192, dtype=np.float32)
    emb, seen = _embedder([want])
    got = emb.embed(_noise(30.0), SR)
    assert len(seen) == 1
    np.testing.assert_array_equal(got, want)


def test_at_the_cap_exactly_it_is_still_one_call() -> None:
    emb, seen = _embedder()
    emb.embed(_noise(SHERPA_MAX_EMBED_S), SR)
    assert len(seen) == 1


def test_past_the_cap_every_window_fits_and_none_is_a_sliver() -> None:
    """Equal windows, not cap-plus-leftover.

    125 s split as 120 + 5 would hand five seconds the same vote as two full
    minutes of speech, so the split is even: two windows of 62.5 s.
    """
    emb, seen = _embedder()
    emb.embed(_noise(125.0), SR)
    assert len(seen) == 2
    assert max(seen) <= SHERPA_MAX_EMBED_S * SR
    assert min(seen) > 60.0 * SR  # no sliver
    assert sum(seen) == int(125.0 * SR)  # the whole buffer is used, once


@pytest.mark.parametrize(
    ("seconds", "windows"),
    [(240.0, 2), (361.0, 4), (1800.0, 15), (3600.0, 30)],
)
def test_the_split_is_the_fewest_windows_that_fit(seconds: float, windows: int) -> None:
    """Fewer and longer windows move the answer less, so the split takes the minimum."""
    emb, seen = _embedder()
    emb.embed(_noise(seconds), SR)
    assert len(seen) == windows
    assert max(seen) <= SHERPA_MAX_EMBED_S * SR


def test_pooling_is_a_direction_and_the_single_call_scale_is_restored() -> None:
    """Unit-normalise, average, then put the magnitude back.

    Two equal windows pointing along different axes with norms 0.2 and 0.4: the
    direction must be their bisector (magnitudes get no vote, or the louder
    window would quietly win), and the returned magnitude must be the weighted
    mean norm, 0.3 — so a caller that skips its own normalisation sees the same
    scale either side of the cap.
    """
    e1, e2 = np.zeros(192, np.float32), np.zeros(192, np.float32)
    e1[0], e2[1] = 0.2, 0.4
    emb, seen = _embedder([e1, e2])
    got = emb.embed(_noise(240.0), SR)

    assert len(seen) == 2 and seen[0] == seen[1]
    assert np.isclose(np.linalg.norm(got), 0.3, atol=1e-5)
    assert np.isclose(got[0], got[1], atol=1e-6)
    np.testing.assert_allclose(got[2:], 0.0, atol=1e-7)


def test_the_even_split_gives_every_window_the_same_vote() -> None:
    """No window outweighs another, because the split makes them the same length.

    Stated as a test because the pooling is sample-weighted, which would read as
    if some window could dominate. It cannot: that weighting is what keeps the
    even split honest, not a knob.
    """
    e1, e2 = np.zeros(192, np.float32), np.zeros(192, np.float32)
    e1[0], e2[1] = 1.0, 1.0
    emb, seen = _embedder([e1, e2])
    got = emb.embed(_noise(200.0), SR)
    assert len(seen) == 2
    assert abs(seen[0] - seen[1]) <= 1  # equal to the sample
    assert np.isclose(got[0], got[1], atol=1e-6)  # so equal vote


def test_one_bad_window_costs_its_share_not_the_whole_voiceprint() -> None:
    """A speaker with 30 minutes of audio must not be erased by one failing window."""
    good = np.zeros(192, np.float32)
    good[5] = 1.0
    emb = _SherpaEmbedder(model_path="/nonexistent.onnx")
    calls: list[int] = []

    def flaky(pcm, sr):
        calls.append(int(pcm.shape[0]))
        if len(calls) == 2:
            raise RuntimeError("12288 by 12298")
        return good

    emb._embed_one = flaky  # type: ignore[method-assign]
    got = emb.embed(_noise(500.0), SR)
    assert len(calls) == 5
    assert np.isclose(got[5], 1.0, atol=1e-5)


def test_total_failure_is_raised_not_returned_as_a_zero_vector() -> None:
    """A zero vector is indistinguishable from a quiet speaker once it is cached.

    This failure already passed in silence once, which is why it is loud now.
    """
    emb = _SherpaEmbedder(model_path="/nonexistent.onnx")

    def always_fails(pcm, sr):
        raise RuntimeError("12288 by 12298")

    emb._embed_one = always_fails  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="none of the 3 window"):
        emb.embed(_noise(300.0), SR)


def test_a_degenerate_window_is_skipped_rather_than_poisoning_the_pool() -> None:
    """A zero-norm window contributes nothing instead of dragging the mean to zero."""
    good = np.zeros(192, np.float32)
    good[3] = 0.5
    emb, seen = _embedder([np.zeros(192, np.float32), good])
    got = emb.embed(_noise(240.0), SR)
    assert len(seen) == 2
    assert np.isclose(np.linalg.norm(got), 0.5, atol=1e-5)
    assert np.isclose(got[3], 0.5, atol=1e-5)
