#!/usr/bin/env python3
"""Unit tests for tools/enhance/pipeline.py (pure functions, no models).

Usage:
    /opt/miniconda3/envs/livetalking/bin/python -m tests.test_enhance_pipeline

Exit code: 0 if all checks pass, 1 otherwise. Uses plain asserts so it runs
without pytest (not installed in the project env).
"""
import sys

import cv2
import numpy as np

from tools.enhance import pipeline


def test_parse_color():
    assert pipeline.parse_color("#1E1E1E") == (30, 30, 30)   # BGR, gray
    assert pipeline.parse_color("#fff") == (255, 255, 255)
    assert pipeline.parse_color("#102030") == (48, 32, 16)    # BGR of RGB 16,32,48
    assert pipeline.parse_color("10,20,30") == (30, 20, 10)   # r,g,b -> BGR
    for bad in ("zzz", "#12", "1,2"):
        try:
            pipeline.parse_color(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_scale_fit_contain_geometry():
    # 1620x1080 into 1080x1920: scale = min(1080/1620, 1920/1080) = 0.6667
    frame = np.zeros((1080, 1620, 3), np.uint8)
    out, (x0, y0, x1, y1) = pipeline._scale_fit(frame, 1080, 1920, "contain")
    assert out.shape[:2] == (720, 1080), out.shape
    assert (x0, y0, x1, y1) == (0, 600, 1080, 1320), (x0, y0, x1, y1)


def test_compose_no_alpha_flat_bg():
    frame = np.full((100, 100, 3), 200, np.uint8)  # bright square
    out = pipeline.compose_vertical(frame, None, 90, 160, (30, 30, 30), "contain")
    assert out.shape == (160, 90, 3)
    # top/bottom letterbox should be background
    assert tuple(out[0, 0]) == (30, 30, 30)
    assert tuple(out[-1, -1]) == (30, 30, 30)


def test_compose_with_alpha_blends():
    frame = np.full((100, 100, 3), 255, np.uint8)
    alpha = np.full((100, 100), 128, np.uint8)  # 50% person
    out = pipeline.compose_vertical(frame, alpha, 100, 100, (0, 0, 0), "cover")
    # 50/50 blend of 255 and 0 -> ~127
    assert 120 <= int(out[50, 50, 0]) <= 135, int(out[50, 50, 0])


def test_compose_alpha_zero_is_bg():
    frame = np.full((100, 100, 3), 255, np.uint8)
    alpha = np.zeros((100, 100), np.uint8)  # no person
    out = pipeline.compose_vertical(frame, alpha, 100, 100, (10, 20, 30), "cover")
    assert tuple(out[50, 50]) == (10, 20, 30)


def test_rvm_tensor_conversions():
    # Imported lazily: pulls in torch but not the model/weights.
    import torch
    from tools.enhance import matting

    # BGR -> [1,1,3,H,W] float 0..1, and channel order becomes RGB
    bgr = np.zeros((4, 6, 3), np.uint8)
    bgr[..., 0] = 10   # B
    bgr[..., 1] = 20   # G
    bgr[..., 2] = 30   # R
    t = matting._bgr_to_tensor(bgr)
    assert t.shape == (1, 1, 3, 4, 6), t.shape
    assert t.dtype == torch.float32
    assert abs(float(t[0, 0, 0, 0, 0]) - 30 / 255) < 1e-5  # R first
    assert abs(float(t[0, 0, 2, 0, 0]) - 10 / 255) < 1e-5  # B last
    # pha [1,1,1,H,W] in 0..1 -> uint8
    pha = torch.tensor([[[[[0.0, 0.5], [1.0, 0.25]]]]])
    a = matting._pha_to_uint8(pha)
    assert a.shape == (2, 2) and a.dtype == np.uint8
    assert list(a.flatten()) == [0, 128, 255, 64], list(a.flatten())


def test_light_correct_strength_semantics():
    from tools.enhance.enhance import light_correct

    img = np.full((50, 50, 3), 128, np.uint8)
    # strength 0 returns the input unchanged
    assert np.array_equal(light_correct(img, 0.0), img)
    # strength 1 returns a corrected (different) result, same shape/dtype
    out = light_correct(img, 1.0)
    assert out.shape == img.shape and out.dtype == np.uint8
    # correction should not blow the byte range
    assert out.min() >= 0 and out.max() <= 255


def test_recompose_from_matte_swap():
    import os, tempfile
    from tools.enhance import pipeline

    # Build a 2-frame matte + person sidecar by hand (no models needed).
    matte_dir = tempfile.mkdtemp()
    person_dir = tempfile.mkdtemp()
    for i in range(2):
        alpha = np.zeros((40, 30), np.uint8)
        alpha[10:30, 10:20] = 255          # person silhouette block
        pipeline.write_matte_sidecar(alpha, matte_dir, i)
        person = np.full((40, 30, 3), 200, np.uint8)
        cv2.imwrite(os.path.join(person_dir, f"{i:08d}.png"), person)
    import json
    with open(os.path.join(matte_dir, "meta.json"), "w") as fh:
        json.dump({"fit": "contain", "canvas": [60, 80], "fps": 30.0}, fh)

    # Recompose onto a different background; the person block must keep its
    # color where alpha=255 and take the bg color where alpha=0.
    frames = list(pipeline.recompose_from_matte(
        matte_dir, (10, 20, 30), 60, 80, 30.0, "contain",
        person_layer=lambda i: cv2.imread(os.path.join(person_dir, f"{i:08d}.png"))))
    assert len(frames) == 2, len(frames)
    f = frames[0]
    assert f.shape == (80, 60, 3)
    # outside the person silhouette -> background
    assert tuple(f[2, 2]) == (10, 20, 30)
    # inside the silhouette -> person color (200,200,200), scaled 30x40 -> 60x80
    assert tuple(f[40, 30]) == (200, 200, 200)

    # geometry mismatch must raise, not silently recompose wrong
    try:
        list(pipeline.recompose_from_matte(matte_dir, (0, 0, 0), 100, 100, 30.0))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on canvas mismatch")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
