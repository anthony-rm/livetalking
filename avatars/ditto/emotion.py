"""Ditto's 8-class emotion vocabulary, and parsing for the API surface.

Ditto conditions the motion model on a one-hot emotion vector: upstream's
``ConditionHandler`` concatenates ``aud_feat[1024] + emo[8] + eye_open[2] +
eye_ball[6] + sc[63]`` and its ``_get_emo_avg`` docstring fixes the label order
below. The index IS the position in ``EMOTIONS``, so an API that accepts either
a name or a bare index has one source of truth.

Kept out of ``ditto_avatar.py`` for the same reason ``artifact.py`` is: it must
be importable and testable on a machine with no weights, no onnxruntime and no
GPU. Nothing here touches torch.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

#: Label order of Ditto's 8-way emotion conditioning (upstream
#: ``ConditionHandler._get_emo_avg``). Neutral (index 4) is the cfg default.
EMOTIONS = ("angry", "disgust", "fear", "happy",
            "neutral", "sad", "surprise", "contempt")

EMOTION_INDEX = {name: i for i, name in enumerate(EMOTIONS)}

#: What ``ConditionHandler.__call__(..., emo=)`` accepts: a label, a list of
#: labels (averaged into one soft target by upstream), or None for the cfg
#: default.
EmotionValue = Union[None, int, List[int]]


def parse_emotion_one(value) -> int:
    """One emotion -> its index. Accepts a name (case-insensitive) or 0..7."""
    # bool is an int subclass; True would silently mean "disgust".
    if isinstance(value, bool):
        raise ValueError(f"emotion must be a name or index, got {value!r}")
    if isinstance(value, str):
        key = value.strip().lower()
        if key in EMOTION_INDEX:
            return EMOTION_INDEX[key]
        raise ValueError(
            f"unknown emotion {value!r}; expected one of {', '.join(EMOTIONS)}")
    if isinstance(value, (int, np.integer)):
        # Upstream's _parse_emo_seq does isinstance(emo, int), which a numpy
        # integer fails — so normalise to a plain int here, not at the model.
        idx = int(value)
        if 0 <= idx < len(EMOTIONS):
            return idx
        raise ValueError(
            f"emotion index {idx} out of range 0..{len(EMOTIONS) - 1}")
    raise ValueError(f"emotion must be a name or index, got {value!r}")


def parse_emotion(value) -> EmotionValue:
    """API value -> what the model takes.

    ``None`` passes through (the cfg default). A list is kept as a list of
    indices, which upstream averages into one soft target — that is how
    "happy+sad" expresses a mixed state rather than picking one.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("emotion list is empty")
        return [parse_emotion_one(v) for v in value]
    return parse_emotion_one(value)


def emotion_name(value: Optional[EmotionValue]) -> str:
    """Canonical, human-readable form for API responses and logs."""
    if value is None:
        return "default"
    if isinstance(value, list):
        return "+".join(EMOTIONS[int(v)] for v in value)
    return EMOTIONS[int(value)]
