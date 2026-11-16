#!/usr/bin/env python3
"""Tests for Ditto's emotion vocabulary and API-level parsing.

No weights, no GPU: the label order is upstream's contract
(``ConditionHandler._get_emo_avg``), and the parser is what the HTTP layer
feeds — a wrong name or an out-of-range index must fail LOUD at the route,
not produce a garbage conditioning vector.
"""
from __future__ import annotations

import numpy as np
import pytest

from avatars.ditto.emotion import (
    EMOTIONS, EMOTION_INDEX, emotion_name, parse_emotion,
)

from avatars.ditto_avatar import DittoReal


def test_label_order_is_upstreams():
    """Positions are baked into the cfg's one-hot: happy=3, neutral=4."""
    assert EMOTIONS == ("angry", "disgust", "fear", "happy",
                        "neutral", "sad", "surprise", "contempt")
    assert EMOTION_INDEX["happy"] == 3
    assert EMOTION_INDEX["neutral"] == 4
    assert EMOTION_INDEX["surprise"] == 6


def test_parse_accepts_name_case_insensitive():
    assert parse_emotion("Happy") == 3
    assert parse_emotion("  SURPRISE ") == 6


def test_parse_accepts_int_and_numpy_int():
    assert parse_emotion(2) == 2
    assert parse_emotion(np.int64(2)) == 2


def test_parse_none_passes_through():
    """None means "no override" — the cfg default stays in force."""
    assert parse_emotion(None) is None


def test_parse_list_keeps_list_form():
    """Upstream averages a label list into one soft target."""
    assert parse_emotion([3, 4]) == [3, 4]
    assert parse_emotion(("happy", "sad")) == [3, 5]


def test_parse_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown emotion"):
        parse_emotion("excited")


def test_parse_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="out of range"):
        parse_emotion(8)
    with pytest.raises(ValueError, match="out of range"):
        parse_emotion(-1)


def test_parse_rejects_bool():
    """bool is an int subclass; True would silently mean "disgust"."""
    with pytest.raises(ValueError):
        parse_emotion(True)


def test_emotion_name_roundtrip():
    assert emotion_name(3) == "happy"
    assert emotion_name(None) == "default"
    assert emotion_name([3, 5]) == "happy+sad"


# ─── backend window emotion ────────────────────────────────────────────────
def make_backend(override=None):
    obj = object.__new__(DittoReal)
    obj._emo_override = override
    return obj


def test_window_emotion_takes_the_mode():
    """A window spanning an utterance boundary resolves as its most common
    tag, not whichever came first or last."""
    b = make_backend(override=None)
    assert b._window_emotion([None, 3, 3, None]) == 3
    assert b._window_emotion([3, None, None, None]) == 3
    assert b._window_emotion([None, None, None, None]) is None


def test_window_emotion_untagged_falls_back_to_sticky_override():
    b = make_backend(override=3)
    assert b._window_emotion([None] * 4) == 3
    # A tagged window wins over the sticky override.
    assert b._window_emotion([None, 5, 5, None]) == 5


def test_window_emotion_mixed_tags_mode_wins():
    b = make_backend()
    assert b._window_emotion([3, 3, 5, None]) == 3
    assert b._window_emotion([5, 3, 3, 3]) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
