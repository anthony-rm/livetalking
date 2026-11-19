"""Unit tests for the emotion lab's motion math (no GPU, no weights).

`exaggerate_pose` is the knob that actually changes how strong an emotion
reads, and it is pure numpy — so it must be provable off-box:

* gain 1 is a no-op (the default path must not perturb the render)
* the amplified value equals neutral + gain * delta, per pose channel
* `exp` is never touched (Ditto's emotion signal is pose, and editing exp
  invents a facial expression the model never produced)
* a bone-straight raise of the pose deltas cannot silently change shape
"""
import numpy as np
import pytest

from scripts.ditto.emotion_lab import (MOTION_GROUPS, _arr2dic, _motion_spans,
                                       exaggerate_pose)

# Deterministic pseudo-motion: 12 frames of 265 channels.
RNG = np.random.RandomState(7)
MOTION = RNG.randn(12, 265).astype(np.float32) * 0.05
NEUTRAL = RNG.randn(12, 265).astype(np.float32) * 0.05

POSE = ("pitch", "yaw", "roll", "t")


def span(name):
    for n, s, e in _motion_spans():
        if n == name:
            return s, e
    raise KeyError(name)


def test_channel_layout_covers_the_motion_vector():
    """265 is upstream's dim; a wrong layout would silently shift every group."""
    assert sum(n for _, n in MOTION_GROUPS) == 265
    assert len(list(_motion_spans())) == len(MOTION_GROUPS)


def test_gain_one_is_a_no_op():
    out = exaggerate_pose(MOTION, NEUTRAL, 1.0)
    np.testing.assert_array_equal(out, MOTION)
    assert out is MOTION          # not just equal: no copy is even needed


def test_gain_one_is_a_no_op_without_a_neutral_reference():
    np.testing.assert_array_equal(exaggerate_pose(MOTION, None, 2.0), MOTION)


@pytest.mark.parametrize("gain", [0.5, 2.0, 3.5])
def test_pose_groups_follow_neutral_plus_gain_times_delta(gain):
    out = exaggerate_pose(MOTION, NEUTRAL, gain)
    for name in POSE:
        s, e = span(name)
        expected = NEUTRAL[:, s:e] + gain * (MOTION[:, s:e] - NEUTRAL[:, s:e])
        np.testing.assert_allclose(out[:, s:e], expected, rtol=1e-6, atol=1e-7)


def test_exp_is_never_touched():
    """The facial-expression channels are left at the model's own value."""
    out = exaggerate_pose(MOTION, NEUTRAL, 4.0)
    s, e = span("exp")
    np.testing.assert_array_equal(out[:, s:e], MOTION[:, s:e])


def test_scale_is_never_touched():
    """scale-1 is stored as an offset; amplifying it would move the avatar's
    apparent size, which is not an emotion."""
    out = exaggerate_pose(MOTION, NEUTRAL, 4.0)
    s, e = span("scale")
    np.testing.assert_array_equal(out[:, s:e], MOTION[:, s:e])


def test_gain_zero_returns_the_neutral_pose():
    """gain 0 = the label contributes no pose at all."""
    out = exaggerate_pose(MOTION, NEUTRAL, 0.0)
    for name in POSE:
        s, e = span(name)
        np.testing.assert_allclose(out[:, s:e], NEUTRAL[:, s:e], atol=1e-7)


def test_the_original_motion_is_not_mutated():
    """Sampling is expensive; a caller may reuse the array for another gain."""
    before = MOTION.copy()
    exaggerate_pose(MOTION, NEUTRAL, 3.0)
    np.testing.assert_array_equal(MOTION, before)


def test_output_keeps_shape_and_dtype():
    out = exaggerate_pose(MOTION, NEUTRAL, 2.0)
    assert out.shape == MOTION.shape
    assert out.dtype == MOTION.dtype


def test_custom_groups_are_honoured():
    out = exaggerate_pose(MOTION, NEUTRAL, 3.0, groups=("yaw",))
    s, e = span("yaw")
    expected = NEUTRAL[:, s:e] + 3.0 * (MOTION[:, s:e] - NEUTRAL[:, s:e])
    np.testing.assert_allclose(out[:, s:e], expected, atol=1e-7)
    s, e = span("pitch")
    np.testing.assert_array_equal(out[:, s:e], MOTION[:, s:e])


def test_arr2dic_undoes_the_scale_offset():
    """Upstream stores scale-1 on the wire; stitch expects scale back."""
    vec = np.zeros(265, np.float32)
    dic = _arr2dic(vec)
    assert set(dic) == {n for n, _ in MOTION_GROUPS}
    assert dic["scale"][0, 0] == pytest.approx(1.0)
    assert dic["pitch"].shape == (1, 66)
    assert dic["t"].shape == (1, 3)
