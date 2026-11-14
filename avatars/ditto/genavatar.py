"""Ditto artifact writer — video/image -> data/avatars/<id>/ ditto layout.

Bridges from the app's genavatar contract (generate_avatar(video_path,
avatar_id, ...)) to upstream's AvatarRegistrar, then writes the layout the
backend reads (avatars/ditto/artifact.py). Also returns the info dict the
backend would need to warm up, mirroring musetalk's tuple returns.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from typing import Callable, Optional

import cv2
import numpy as np

from avatars.ditto import artifact as art
from utils.logger import logger

DITTO_HOME = os.environ.get("DITTO_HOME", os.path.expanduser("~/ditto-probe"))
DITTO_WEIGHTS = os.environ.get("DITTO_WEIGHTS",
                               os.path.expanduser("~/ditto-weights/ditto_pytorch"))
DITTO_CFG = os.environ.get("DITTO_CFG",
                           os.path.expanduser(
                               "~/ditto-weights/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"))


def _ensure():
    core = os.path.join(DITTO_HOME, "core")
    if DITTO_HOME not in sys.path:
        sys.path.insert(0, DITTO_HOME)


def generate_avatar(video_path: str, avatar_id: str,
                    save_path: str = "./data/avatars",
                    max_frames: int = 75,
                    progress_callback: Optional[Callable[[int], None]] = None) -> str:
    """Register a source video/image and write the ditto artifact.

    max_frames caps the registered clip (f_s is 8.4 MB/frame fp32 -> 4.2 fp16,
    so a 75-frame cap keeps the artifact near 315 MB).
    """
    _ensure()
    from core.atomic_components.cfg import parse_cfg
    from core.atomic_components.avatar_registrar import (
        AvatarRegistrar, smooth_x_s_info_lst)

    avatar_dir = os.path.join(save_path, avatar_id)
    os.makedirs(avatar_dir, exist_ok=True)

    if progress_callback:
        progress_callback(5)

    (reg_cfg, _cond, _lmdm, _stitch, _warp, _dec, _w2f, dkw) = parse_cfg(
        DITTO_CFG, DITTO_WEIGHTS)
    # Aux ORT models on CPU (cfg device keys), registration is not VRAM-critical
    registrar = AvatarRegistrar(**reg_cfg)

    if progress_callback:
        progress_callback(10)

    t0 = time.time()
    source_info = registrar.register(video_path, max_dim=dkw["max_size"],
                                     n_frames=max_frames,
                                     crop_scale=dkw["crop_scale"],
                                     crop_vx_ratio=dkw["crop_vx_ratio"],
                                     crop_vy_ratio=dkw["crop_vy_ratio"],
                                     crop_flag_do_rot=dkw["crop_flag_do_rot"])
    logger.info("[ditto-genavatar] registered %d frames in %.1fs",
                len(source_info["x_s_info_lst"]), time.time() - t0)

    if progress_callback:
        progress_callback(70)

    if len(source_info["x_s_info_lst"]) > 1 and dkw["smo_k_s"] > 1:
        source_info["x_s_info_lst"] = smooth_x_s_info_lst(
            source_info["x_s_info_lst"], smo_k=dkw["smo_k_s"])

    n = len(source_info["x_s_info_lst"])
    render_size_of_f_s = int(np.asarray(source_info["f_s_lst"][0]).shape[-1]) * 8  # 64->512

    # ── write the artifact ──
    # cv2.imwrite never creates directories; every subdir must exist first.
    os.makedirs(os.path.join(avatar_dir, art.FULL_IMGS_DIR), exist_ok=True)
    for i in range(n):
        f_s_i = np.asarray(source_info["f_s_lst"][i])
        # registration returns a batch-dim volume (1,32,16,64,64); the artifact
        # stores the per-frame (32,16,64,64) and the reader re-adds [None].
        if f_s_i.ndim == 5:
            f_s_i = f_s_i[0]
        art.write_f_s(avatar_dir, i, f_s_i)
        art.write_x_s_info(avatar_dir, i, source_info["x_s_info_lst"][i])
        img = np.asarray(source_info["img_rgb_lst"][i])
        ok = cv2.imwrite(art.full_img_path(avatar_dir, i),
                         cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(
                f"cv2.imwrite failed for full_imgs/{i:08d}.png "
                f"(img {img.shape} {img.dtype} -> "
                f"{art.full_img_path(avatar_dir, i)})")
        if progress_callback:
            progress_callback(70 + int(20 * (i + 1) / n))

    with open(os.path.join(avatar_dir, "M_c2o.pkl"), "wb") as f:
        pickle.dump([np.asarray(m, dtype=np.float32)
                     for m in source_info["M_c2o_lst"]], f)

    art.write_marker(avatar_dir, art.build_marker(
        avatar_id=avatar_id, frame_count=n,
        render_size=render_size_of_f_s, has_eye_state=True,
        source=os.path.abspath(video_path)))

    if progress_callback:
        progress_callback(100)
    logger.info("[ditto-genavatar] wrote %d frames to %s", n, avatar_dir)
    return avatar_dir
