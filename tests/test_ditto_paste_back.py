#!/usr/bin/env python3
"""Paste-back contract tests for the Ditto backend.

No weights, no GPU, no DITTO_HOME: the methods under test only need the
artifact geometry and the mask, so the instance is built with __new__ and the
few attributes paste_back_frame reads are set directly.

The colour test exists because the decoder returns RGB while the artifact
frames are cv2.imread BGR; blending them without a conversion produced the
blue-face bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from avatars.ditto_avatar import DittoReal


def make_backend(frame_bgr, M_c2o, mask_value=1.0):
    obj = object.__new__(DittoReal)
    obj.frame_list_cycle = [frame_bgr]
    obj.M_c2o_list = [np.asarray(M_c2o, dtype=np.float32)]
    obj._paste_geom_cache = {}
    h, w = frame_bgr.shape[:2]
    obj._mask_cache = np.full((h, w, 3), mask_value, dtype=np.float32)
    return obj


IDENTITY = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


def test_rgb_render_lands_as_bgr_pixels():
    frame = np.zeros((64, 64, 3), np.uint8)          # BGR black
    backend = make_backend(frame, IDENTITY)
    render_rgb = np.zeros((64, 64, 3), np.uint8)
    render_rgb[..., 0] = 255                        # pure RED in RGB order

    out = backend.paste_back_frame(render_rgb, 0)

    # ... which is BGR (0, 0, 255) once pasted into a BGR frame.
    assert tuple(out[32, 32]) == (0, 0, 255)


def test_blend_keeps_channels_separate():
    frame = np.full((64, 64, 3), 200, np.uint8)      # BGR grey
    backend = make_backend(frame, IDENTITY, mask_value=0.0)
    render_rgb = np.zeros((64, 64, 3), np.uint8)
    render_rgb[..., 0] = 255

    out = backend.paste_back_frame(render_rgb, 0)    # mask 0 -> source frame

    assert tuple(out[32, 32]) == (200, 200, 200)


def test_none_render_returns_a_copy_of_the_source_frame():
    frame = np.zeros((64, 64, 3), np.uint8)
    backend = make_backend(frame, IDENTITY)

    out = backend.paste_back_frame(None, 0)

    assert np.array_equal(out, frame)
    assert out is not frame


def test_paste_back_does_not_mutate_the_artifact_matrix():
    """A second render size reuses the same M_c2o; the bbox shift must not
    accumulate into the artifact's own matrix."""
    frame = np.zeros((128, 128, 3), np.uint8)
    backend = make_backend(frame, IDENTITY)
    before = backend.M_c2o_list[0].copy()

    for size in (64, 32):
        backend._mask_cache = np.full((size, size, 1), 1.0, np.float32)
        backend.paste_back_frame(np.zeros((size, size, 3), np.uint8), 0)

    assert np.array_equal(backend.M_c2o_list[0], before)


def test_crop_matrix_is_cached_per_render_size():
    frame = np.zeros((128, 128, 3), np.uint8)
    backend = make_backend(frame, IDENTITY)

    first = backend._paste_geom(0, 64, 64)
    again = backend._paste_geom(0, 64, 64)
    other = backend._paste_geom(0, 32, 32)

    assert first is again
    assert first is not other
    assert first[1] == (0, 64, 0, 64)
    assert other[1] == (0, 32, 0, 32)


def test_translation_is_offset_by_the_bbox_origin():
    frame = np.zeros((128, 128, 3), np.uint8)
    shift = np.array([[1, 0, 40], [0, 1, 20], [0, 0, 1]], dtype=np.float32)
    backend = make_backend(frame, shift)

    Mc, bbox, _, _ = backend._paste_geom(0, 32, 32)

    assert bbox == (40, 72, 20, 52)
    assert Mc[0, 2] == 0 and Mc[1, 2] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
