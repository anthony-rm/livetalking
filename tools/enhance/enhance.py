"""Quality enhancement: lighting + Real-ESRGAN super-res + GFPGAN face restore.

API contracts verified against the installed packages:
- `RealESRGANer(scale, model_path, tile, tile_pad, pre_pad, half, device).enhance(img, outscale)`
  -> `(output_bgr_uint8, img_mode)`. img is BGR uint8; output scaled by `outscale`.
- `GFPGANer(model_path, upscale, arch, channel_multiplier, bg_upsampler, device).enhance(img, weight)`
  -> `(cropped_faces, restored_faces, restored_img)`. It detects faces (dlib),
  restores each on a 512 crop, and pastes back. When `bg_upsampler` (a
  RealESRGANer) is given it also upscales the background, so one call does SR +
  face restore at a uniform scale.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
# Weight URLs live in scripts/download_enhance_weights.py (kept there so the
# downloader and this module don't drift).
DEFAULT_UPSCALE = 1


def _device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def _weight_path(name: str) -> str:
    path = os.path.join(MODELS_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing weights {path}. Run scripts/download_enhance_weights.py "
            f"(or set {name} in models/) before enhancing.")
    return path


# --------------------------------------------------------------------------- #
# Lighting correction (classical, cheap, same-resolution)
# --------------------------------------------------------------------------- #
def light_correct(frame_bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Auto white-balance (gray-world) + CLAHE on the luminance channel.

    `strength` in 0..1 blends the corrected result toward the original, so 0
    leaves the frame untouched and 1 applies the full correction.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0:
        return frame_bgr

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # gray-world WB: scale a/b channels so their means match a neutral gray
    a_mean, b_mean = a.mean(), b.mean()
    target = 128.0
    a_gain = target / (a_mean + 1e-6)
    b_gain = target / (b_mean + 1e-6)
    a = np.clip(a * a_gain, 0, 255)
    b = np.clip(b * b_gain, 0, 255)

    # CLAHE contrast on L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l.astype(np.uint8)).astype(np.float32)

    corrected = cv2.merge([l, a, b]).astype(np.uint8)
    corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    if strength == 1.0:
        return corrected
    return cv2.addWeighted(corrected, strength, frame_bgr, 1 - strength, 0)


# --------------------------------------------------------------------------- #
# SR + face restore
# --------------------------------------------------------------------------- #
class Enhancer:
    """Owns the Real-ESRGAN + GFPGAN models and runs the enhancement pipeline."""

    def __init__(self, upscale: int = DEFAULT_UPSCALE, face_restore: bool = True,
                 tile: int = 400, device: str = "auto") -> None:
        self.upscale = upscale
        self.face_restore = face_restore
        self.device = _device(device)

        self.sr = None
        self.gfpgan = None
        if upscale > 1:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=upscale)
            self.sr = RealESRGANer(scale=upscale, model_path=_weight_path("RealESRGAN_x4plus.pth"),
                                   model=sr_model, tile=tile, tile_pad=10,
                                   pre_pad=0, half=False, device=self.device)
        if face_restore:
            from gfpgan import GFPGANer
            self.gfpgan = GFPGANer(
                model_path=_weight_path("GFPGANv1.4.pth"),
                upscale=max(upscale, 1), arch="clean", channel_multiplier=2,
                bg_upsampler=self.sr, device=self.device)

    def enhance(self, frame_bgr: np.ndarray, strength: float = 0.5) -> Tuple[np.ndarray, int]:
        """Lighting -> (SR+face restore). Returns (enhanced BGR, scale factor)."""
        frame = light_correct(frame_bgr, strength)

        if self.gfpgan is not None:
            try:
                _, _, restored = self.gfpgan.enhance(frame, weight=float(strength))
                if restored is not None:
                    return restored, max(self.upscale, 1)
            except RuntimeError as exc:
                print(f"  GFPGAN failed ({exc}); falling back to SR-only")
        if self.sr is not None:
            out, _ = self.sr.enhance(frame, outscale=self.upscale)
            return out, self.upscale
        return frame, 1


# Lazy module-level default, mirroring matting.get_default_runner().
_default: Optional[Enhancer] = None


def get_default_enhancer(**kw) -> Enhancer:
    global _default
    if _default is None:
        _default = Enhancer(**kw)
    return _default


def reset_default_enhancer() -> None:
    global _default
    _default = None