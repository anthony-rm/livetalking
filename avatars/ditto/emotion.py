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


def one_hot_vectors(indices: List[int], scale: float = 8.0) -> np.ndarray:
    """Upstream's ``_get_emo_avg``: one-hot * scale, softmaxed.

    Reproduced here (rather than imported) because it lives in the vendored
    Ditto core, which is only on the GPU box. ``scale=8`` makes a single label
    99.77% one-hot, so a label is already near the model's maximum input
    amplitude — there is no "louder" one-hot to reach for.

    Softmax is written out (upstream uses ``scipy.special.softmax``) because
    this module loads on machines with no scipy: it is a three-line function
    and the dependency would buy nothing.
    """
    logits = np.zeros(len(EMOTIONS), dtype=np.float32)
    for i in indices:
        logits[int(i)] = scale
    e = np.exp(logits - logits.max())
    return (e / e.sum()).astype(np.float32)


def emotion_vector(value: EmotionValue, gain: float = 1.0) -> np.ndarray:
    """The ``(1, 8)`` conditioning vector for ``value``, optionally amplified.

    ``gain`` steers along the neutral->label axis, in the softmax's own
    probability space:

        gain 0 -> the neutral vector (label has no effect at all)
        gain 1 -> exactly what upstream feeds the model (unchanged)
        gain 2 -> twice the distance from neutral

    This is the honest place to turn emotion up. The one-hot itself has no
    headroom (see ``one_hot_vectors``), whereas the effect a label has on the
    render is proportional to how far its vector sits from neutral — doubling
    that distance asks the motion model for a stronger reading of the same
    label without inventing a ninth class.

    Negative gain flips the label (fear at -1 is the mirror image of fear), so
    values below 0 are rejected: that is a different experiment, not a louder
    one.
    """
    if value is None:
        raise ValueError("emotion_vector needs a concrete emotion, not None")
    if gain < 0:
        raise ValueError(f"emotion gain must be >= 0, got {gain}")
    idxs = ([int(v) for v in value] if isinstance(value, (list, tuple))
            else [int(value)])
    target = one_hot_vectors(idxs)
    if gain == 1.0:
        return target.reshape(1, 8)
    neutral = one_hot_vectors([EMOTION_INDEX["neutral"]])
    return (neutral + gain * (target - neutral)).reshape(1, 8).astype(np.float32)
