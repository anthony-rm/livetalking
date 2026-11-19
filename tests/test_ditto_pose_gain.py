"""Serving-path pose exaggeration: `_exaggerate` on a real-shaped array.

The lab proves the maths; these prove the wiring. The properties that matter:

* gain 1 is a no-op — the default render must be untouched, byte for byte
* only the pose span moves; `scale` and `exp` are preserved exactly
* the source pose (`a2m.s_kp_cond`) is the reference, so no neutral render and
  no extra LMDM pass is needed
* a missing/short source pose degrades to no-op instead of raising inside the
  inference thread
"""
import numpy as np
import pytest

import avatars.ditto_avatar as da

POSE_SPAN = (1, 202)      # scale(1) | pitch/yaw/roll(198) | t(3)
MOTION_DIM = 265          # + exp(63)


class FakeA2M:
    """Only the two attributes `_exaggerate` reads."""

    def __init__(self, src_width=265, missing=False):
        if missing:
            return
        rng = np.random.RandomState(3)
        self.s_kp_cond = rng.randn(1, src_width).astype(np.float32) * 0.1


def make_backend(gain=1.0, **kwargs):
    """A DittoReal with just the state `_exaggerate` touches."""
    obj = object.__new__(da.DittoReal)
    obj.a2m = FakeA2M(**kwargs)
    obj._emo_gain = gain
    return obj


def motion(n=5, seed=11):
    return (np.random.RandomState(seed).randn(n, MOTION_DIM)
            .astype(np.float32) * 0.2)


def test_gain_one_returns_the_array_untouched():
    b = make_backend(gain=1.0)
    raw = motion()
    out = b._exaggerate(raw)
    np.testing.assert_array_equal(out, raw)
    assert out is raw          # not even a copy: the hot path stays hot


@pytest.mark.parametrize("gain", [0.5, 1.5, 2.0, 3.0])
def test_pose_grows_by_the_gain_relative_to_the_source_pose(gain):
    b = make_backend(gain=gain)
    raw = motion()
    out = b._exaggerate(raw)
    s, e = POSE_SPAN
    ref = np.asarray(b.a2m.s_kp_cond, np.float32).reshape(1, -1)[:, s:e]
    expected = ref + gain * (raw[:, s:e] - ref)
    np.testing.assert_allclose(out[:, s:e], expected, rtol=1e-5, atol=1e-6)


def test_scale_and_exp_are_never_touched():
    """scale is the avatar's apparent size, exp its face: neither is a pose."""
    b = make_backend(gain=3.0)
    raw = motion()
    out = b._exaggerate(raw)
    s, e = POSE_SPAN
    np.testing.assert_array_equal(out[:, :s], raw[:, :s])      # scale
    np.testing.assert_array_equal(out[:, e:], raw[:, e:])      # exp


def test_the_delta_is_linear_so_gain_is_predictable():
    """Pose gain composes: applying 2x twice equals applying 4x once."""
    raw = motion()
    once = make_backend(gain=4.0)._exaggerate(raw)
    twice = make_backend(gain=2.0)._exaggerate(
        make_backend(gain=2.0)._exaggerate(raw))
    s, e = POSE_SPAN
    np.testing.assert_allclose(twice[:, s:e], once[:, s:e], atol=1e-5)


def test_the_input_array_is_not_mutated():
    """The same window is also consumed by cvt_fmt; corrupting it would be
    silent and would poison the following frames."""
    b = make_backend(gain=2.0)
    raw = motion()
    before = raw.copy()
    b._exaggerate(raw)
    np.testing.assert_array_equal(raw, before)


def test_a_missing_source_pose_is_a_no_op():
    """Older artifacts and the TRT path may not expose s_kp_cond. Degrading to
    the model's own motion is correct; raising inside the inference thread is
    not."""
    b = make_backend(gain=3.0, missing=True)
    b.a2m.s_kp_cond = None
    raw = motion()
    np.testing.assert_array_equal(b._exaggerate(raw), raw)


def test_the_source_pose_is_full_width_and_the_span_stays_fixed():
    """The real `s_kp_cond` is 265 wide — the whole vector. A span derived from
    its width would sweep `exp` in and deform the face, so the span must be
    fixed at the pose channels regardless."""
    b = make_backend(src_width=265)
    assert np.asarray(b.a2m.s_kp_cond).shape == (1, 265)
    assert b._pose_span() == POSE_SPAN
    raw = motion()
    out = b._exaggerate(raw)
    s, e = POSE_SPAN
    np.testing.assert_array_equal(out[:, e:], raw[:, e:])     # exp survives


def test_a_short_source_pose_is_still_a_no_op_for_exp():
    """Even a source pose shorter than the pose span must not move `exp`."""
    b = make_backend(gain=3.0, src_width=202)
    raw = motion()
    out = b._exaggerate(raw)
    s, e = POSE_SPAN
    np.testing.assert_array_equal(out[:, e:], raw[:, e:])


def test_pose_span_covers_exactly_the_pose_channels():
    b = make_backend()
    assert b._pose_span() == POSE_SPAN
    s, e = POSE_SPAN
    assert e - s == 66 * 3 + 3          # pitch+yaw+roll + t
    assert MOTION_DIM - e == 63         # exp
