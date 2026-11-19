"""The emotion vector: upstream's one-hot, and the gain axis built on it.

These assert the two properties the exaggeration feature rests on: gain=1 is
byte-identical to what upstream feeds the model, and gain is a straight line
away from neutral in probability space.
"""
import numpy as np
import pytest

from avatars.ditto import emotion as de


def test_one_hot_matches_upstream_scale():
    """_get_emo_avg is softmax(one_hot * 8): 99.77% on the label."""
    v = de.one_hot_vectors([de.EMOTION_INDEX["happy"]])
    assert v.shape == (8,)
    assert v.sum() == pytest.approx(1.0, abs=1e-6)
    assert v[3] == pytest.approx(0.9977, abs=1e-3)
    others = np.delete(v, 3)
    assert others.max() < 1e-3


def test_one_hot_averages_a_list():
    """Two labels share the mass evenly (upstream's list-averaging)."""
    v = de.one_hot_vectors([3, 5])
    assert v[3] == pytest.approx(v[5], abs=1e-7)
    assert v[3] > 0.49


def test_neutral_is_not_a_zero_vector():
    v = de.one_hot_vectors([de.EMOTION_INDEX["neutral"]])
    assert v.argmax() == 4
    assert v[4] > 0.99


def test_gain_one_is_exactly_upstream():
    """The default path must not perturb what the model already received."""
    for name in de.EMOTIONS:
        idx = de.EMOTION_INDEX[name]
        got = de.emotion_vector(idx, gain=1.0)
        assert got.shape == (1, 8)
        np.testing.assert_array_equal(got[0], de.one_hot_vectors([idx]))


def test_gain_zero_collapses_to_neutral():
    """Every label at gain 0 must land on the SAME vector: neutral."""
    neutral = de.emotion_vector(de.EMOTION_INDEX["neutral"], gain=0.0)
    for name in ("happy", "angry", "fear", "sad"):
        got = de.emotion_vector(de.EMOTION_INDEX[name], gain=0.0)
        np.testing.assert_allclose(got, neutral, atol=1e-7)


def test_gain_is_linear_away_from_neutral():
    """gain 2 = neutral + 2*(target - neutral): the distance doubles."""
    idx = de.EMOTION_INDEX["angry"]
    neutral = de.emotion_vector(de.EMOTION_INDEX["neutral"], gain=0.0)[0]
    one = de.emotion_vector(idx, gain=1.0)[0]
    two = de.emotion_vector(idx, gain=2.0)[0]
    d1 = np.linalg.norm(one - neutral)
    d2 = np.linalg.norm(two - neutral)
    assert d2 == pytest.approx(2.0 * d1, rel=1e-5)


def test_gain_monotonically_strengthens_the_label():
    """More gain must push the label's probability up, not around."""
    idx = de.EMOTION_INDEX["happy"]
    vals = [de.emotion_vector(idx, gain=g)[0][idx] for g in (0.5, 1, 2, 3)]
    assert vals == sorted(vals)


def test_gain_does_not_invent_mass_on_other_labels():
    """Amplification is a two-point axis; it must not leak into other classes.

    The floor is not zero: softmax(one_hot*8) puts ~3.3e-4 on every non-neutral
    label, and the ray from neutral leaves that untouched. What must never
    happen is unrelated mass GROWING with gain.
    """
    idx = de.EMOTION_INDEX["surprise"]
    neutral = de.emotion_vector(de.EMOTION_INDEX["neutral"], gain=0.0)[0]
    for gain in (1.0, 2.0, 3.0):
        got = de.emotion_vector(idx, gain=gain)[0]
        for i in range(8):
            if i in (idx, de.EMOTION_INDEX["neutral"]):
                continue
            np.testing.assert_allclose(got[i], neutral[i], atol=1e-7,
                                       err_msg=de.EMOTIONS[i])


def test_mix_then_gain_stays_a_valid_distribution_at_gain_one():
    """A list is legal conditioning input; averaged it is still a softmax."""
    got = de.emotion_vector([3, 6], gain=1.0)
    assert got.sum() == pytest.approx(1.0, abs=1e-5)


def test_gain_rejects_negative_and_needs_a_concrete_emotion():
    with pytest.raises(ValueError, match=">= 0"):
        de.emotion_vector(3, gain=-1.0)
    with pytest.raises(ValueError, match="concrete emotion"):
        de.emotion_vector(None, gain=1.0)
