#!/usr/bin/env python3
"""Tests for the Ditto artifact format.

These run without Ditto's weights, onnxruntime or a GPU: every failure mode here
is about *data* (device tags, truncation, version drift), and those are exactly
the failures that used to surface hours into a run.

Run with the project interpreter:
    /opt/miniconda3/envs/livetalking/bin/python -m pytest tests/test_ditto_artifact.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from avatars.ditto import artifact as art


F_S_SHAPE = art.F_S_SHAPE   # (32, 16, 64, 64)


def fake_f_s(value=0.0):
    return np.full(F_S_SHAPE, value, dtype=np.float32)


def make_avatar(root, avatar_id="dt", frames=3, render_size=512):
    d = os.path.join(str(root), avatar_id)
    os.makedirs(d, exist_ok=True)
    for i in range(frames):
        art.write_f_s(d, i, fake_f_s(i))
        art.write_x_s_info(d, i, {
            "kp": np.zeros((21, 3), dtype=np.float32) + i,
            "exp": np.zeros((63,), dtype=np.float32),
            "scale": np.array([1.0], dtype=np.float32),
        })
    os.makedirs(os.path.join(d, art.FULL_IMGS_DIR), exist_ok=True)
    for i in range(frames):
        with open(art.full_img_path(d, i), "wb") as fh:
            fh.write(b"\x89PNG fake")
    art.write_marker(d, art.build_marker(
        avatar_id=avatar_id, frame_count=frames, render_size=render_size))
    return d


# ─── round-trip ────────────────────────────────────────────────────────────
def test_write_then_read_f_s_roundtrips(tmp_path):
    d = make_avatar(tmp_path, frames=2)
    arr = art.read_f_s(d, 1)
    assert arr.shape == F_S_SHAPE
    assert arr.dtype == np.float16
    assert float(arr.mean()) == pytest.approx(1.0)


def test_f_s_is_stored_as_fp16_npy_not_pickle(tmp_path):
    """fp16 halves the artifact; .npy (no pickle) keeps the device tag out."""
    d = make_avatar(tmp_path, frames=1)
    path = art.f_s_path(d, 0)
    assert path.endswith(".npy")
    assert np.load(path, allow_pickle=False).dtype == np.float16
    # Roughly half of fp32, plus the npy header.
    raw = os.path.getsize(path)
    assert raw < F_S_SHAPE[0] * F_S_SHAPE[1] * F_S_SHAPE[2] * F_S_SHAPE[3] * 2 + 256


def test_x_s_info_roundtrips_shapes(tmp_path):
    d = make_avatar(tmp_path, frames=1)
    info = art.read_x_s_info(d, 0)
    assert set(info) == {"kp", "exp", "scale"}
    assert info["kp"].size == 63          # 21*3, flattened on write
    assert info["scale"].size == 1


def test_marker_records_the_dtype_and_shape(tmp_path):
    d = make_avatar(tmp_path, frames=2)
    m = art.read_marker(d)
    assert m["backend"] == "ditto"
    assert m["frame_count"] == 2
    assert m["f_s_dtype"] == "float16"
    assert m["f_s_shape"] == list(F_S_SHAPE)


# ─── device safety (the defect this format exists to prevent) ─────────────
def test_to_storable_accepts_cpu_arrays():
    arr = np.zeros(F_S_SHAPE, dtype=np.float32)
    out = art.to_storable(arr, dtype=np.float16)
    assert out.dtype == np.float16
    assert out.shape == F_S_SHAPE


def test_to_storable_strips_a_device_tag():
    """A torch CPU tensor must come out as a plain numpy array."""
    torch = pytest.importorskip("torch")
    t = torch.zeros(*F_S_SHAPE, dtype=torch.float32)     # CPU, but a tensor
    out = art.to_storable(t, dtype=np.float16)
    assert isinstance(out, np.ndarray)
    assert type(out) is np.ndarray                       # not a torch tensor
    assert out.dtype == np.float16


def test_artifact_written_here_loads_without_the_writer(tmp_path):
    """Reader must not need the device the writer used."""
    d = make_avatar(tmp_path, frames=1)
    arr = np.load(art.f_s_path(d, 0), allow_pickle=False)
    assert arr.dtype == np.float16          # loadable anywhere, no CUDA tag
    assert type(arr) is np.ndarray          # a plain array, not a pickled tensor


def test_wrong_shape_is_rejected_at_write(tmp_path):
    d = os.path.join(str(tmp_path), "dt")
    with pytest.raises(art.ArtifactError, match="shape"):
        art.write_f_s(d, 0, np.zeros((1, 256, 64, 64), dtype=np.float32))


def test_wrong_shape_is_rejected_at_read(tmp_path):
    d = make_avatar(tmp_path, frames=1)
    np.save(art.f_s_path(d, 0), np.zeros((8, 8), dtype=np.float16),
            allow_pickle=False)
    with pytest.raises(art.ArtifactError, match="shape"):
        art.read_f_s(d, 0)


# ─── versioning ────────────────────────────────────────────────────────────
def test_newer_version_refuses_to_load(tmp_path):
    """Loading half an artifact silently is worse than refusing."""
    d = make_avatar(tmp_path, frames=1)
    m = art.read_marker(d)
    m["version"] = art.ARTIFACT_VERSION + 1
    art.write_marker(d, m)
    with pytest.raises(art.ArtifactError, match="version"):
        art.read_marker(d)


def test_missing_marker_is_named(tmp_path):
    d = os.path.join(str(tmp_path), "nothing")
    os.makedirs(d)
    with pytest.raises(art.ArtifactError, match="not a ditto artifact"):
        art.read_marker(d)


# ─── truncation detection ─────────────────────────────────────────────────
def test_validate_accepts_a_complete_artifact(tmp_path):
    d = make_avatar(tmp_path, frames=3)
    assert art.validate(d)["frame_count"] == 3


def test_validate_names_the_first_missing_frame(tmp_path):
    """A truncated write must report which frame, not fail at render time."""
    d = make_avatar(tmp_path, frames=3)
    os.remove(art.f_s_path(d, 2))
    with pytest.raises(art.ArtifactError) as exc:
        art.validate(d)
    msg = str(exc.value)
    assert "f_s/" in msg
    assert "2 of 3" in msg
    assert "index 2" in msg


def test_validate_names_a_missing_directory(tmp_path):
    d = make_avatar(tmp_path, frames=1)
    os.remove(art.full_img_path(d, 0))
    with pytest.raises(art.ArtifactError, match="full_imgs/"):
        art.validate(d)


def test_zero_frame_artifact_is_rejected(tmp_path):
    d = make_avatar(tmp_path, frames=1)
    m = art.read_marker(d)
    m["frame_count"] = 0
    art.write_marker(d, m)
    with pytest.raises(art.ArtifactError, match="frame_count"):
        art.validate(d)


# ─── iteration + checksum ─────────────────────────────────────────────────
def test_iter_f_s_yields_frames_in_order(tmp_path):
    d = make_avatar(tmp_path, frames=3)
    values = [float(a.mean()) for a in art.iter_f_s(d)]
    assert values == pytest.approx([0.0, 1.0, 2.0])


def test_checksum_is_stable_and_content_based(tmp_path):
    d1 = make_avatar(tmp_path / "a" if hasattr(tmp_path, "__truediv__") else tmp_path, frames=2)
    a = art.artifact_checksum(d1)
    b = art.artifact_checksum(d1)
    assert a == b
    # A different frame count changes the hash.
    d2 = make_avatar(tmp_path, "dt2", frames=3)
    assert art.artifact_checksum(d2) != a


def test_checksum_detects_a_modified_frame(tmp_path):
    d = make_avatar(tmp_path, frames=2)
    before = art.artifact_checksum(d)
    art.write_f_s(d, 0, fake_f_s(9.0))
    assert art.artifact_checksum(d) != before


# ─── detect.py integration ────────────────────────────────────────────────
def test_detect_recognises_a_written_artifact(tmp_path):
    """The writer and the detector must agree without a hand-built fixture."""
    from avatars import detect
    make_avatar(tmp_path, "dt", frames=2)
    backend, reason = detect.detect_backend("dt", root=str(tmp_path))
    assert backend == "ditto"
    assert "ditto.json" in reason


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
