"""RVM (Robust Video Matting) person-matte wrapper.

RVM is recurrent: it keeps four hidden states across frames, so a stateless
per-frame `compute_alpha()` cannot work. `RVMRunner` owns the model and the
state; `pipeline.compute_alpha(frame, runner)` is the stateless seam the CLI
calls, and it lazily creates a default runner when none is passed.

Model + weights come from torch.hub (there is no pip package for RVM):
    torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True)
which downloads rvm_mobilenetv3.pth into ~/.cache/torch/hub/checkpoints/.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

_HUB_REPO = "PeterL1n/RobustVideoMatting"
_HUB_MODEL = "mobilenetv3"
_DEFAULT_DOWNSAMPLE = 0.5  # largest side -> ~512px for the encoder


def _device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def _bgr_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HxW -> [1, 1, 3, H, W] float32 in 0..1."""
    rgb = frame_bgr[:, :, ::-1].copy()
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div(255.0)
    return t.unsqueeze(0).unsqueeze(0)  # [B=1, T=1, C, H, W]


def _pha_to_uint8(pha: torch.Tensor) -> np.ndarray:
    """[1,1,1,H,W] -> uint8 HxW (255 = person)."""
    a = pha[0, 0, 0].clamp(0, 1).mul(255).round().to(torch.uint8)
    return a.cpu().numpy()


class RVMRunner:
    """Stateful RVM matting runner: one instance per video stream."""

    def __init__(self, variant: str = _HUB_MODEL, device: str = "auto",
                 downsample_ratio: float = _DEFAULT_DOWNSAMPLE) -> None:
        self.variant = variant
        self.device = _device(device)
        self.downsample_ratio = downsample_ratio
        self.rec: List[Optional[torch.Tensor]] = [None] * 4
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        model = torch.hub.load(_HUB_REPO, _HUB_MODEL,
                               trust_repo=True).to(self.device).eval()
        # torch.jit.script+freeze helps on CUDA but is SLOWER and can stall on
        # MPS (measured: mps eager 92ms/frame vs mps jit 162ms/frame). Only
        # script on CUDA; keep the eager module elsewhere.
        if self.device.type == "cuda":
            try:
                model = torch.jit.script(model)
                model = torch.jit.freeze(model)
            except Exception as exc:  # noqa: BLE001
                print(f"  RVM: torch.jit.script failed ({type(exc).__name__}); "
                      f"using eager model")
        return model

    def alpha(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return the person alpha for one BGR frame (uint8 HxW, 255=person)."""
        src = _bgr_to_tensor(frame_bgr).to(self.device)
        with torch.no_grad():
            _, pha, *self.rec = self.model(src, *self.rec, self.downsample_ratio)
        return _pha_to_uint8(pha)

    def reset(self) -> None:
        """Drop recurrent state (start of a new stream)."""
        self.rec = [None] * 4


# Module-level default runner, lazily created by pipeline.compute_alpha().
_default_runner: Optional[RVMRunner] = None


def get_default_runner(device: str = "auto") -> RVMRunner:
    global _default_runner
    if _default_runner is None or _default_runner.device != _device(device):
        _default_runner = RVMRunner(device=device)
    return _default_runner


def reset_default_runner() -> None:
    global _default_runner
    if _default_runner is not None:
        _default_runner.reset()


def write_matte_sidecar(alpha: np.ndarray, out_dir: str, index: int) -> str:
    """Save one alpha frame as a grayscale PNG (255 = person).

    Naming mirrors `full_imgs/00000000.png` so the sidecar is trivially
    consumable by later background-swap code.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{index:08d}.png")
    cv2.imwrite(path, alpha)
    return path


import cv2  # noqa: E402  (kept at bottom so the torch import order is clean)