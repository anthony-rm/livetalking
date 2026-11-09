"""Remote-GPU avatar backend (Option B).

Registers ``@register("avatar", "remote")``. Use with::

    ./run.sh --model remote --remote_worker 192.168.1.42:50051 \\
             --remote_backend musetalk --avatar_id my_avatar

The Mac keeps the whole pipeline (TTS, ASR feature extraction, paste-back,
streaming). Only ``inference_batch`` is offloaded: audio features go to the
Windows/NVIDIA worker, pred crops come back, paste_back runs locally. The model
uses the avatar artifact metadata (frame list, masks, coords) from the Mac's
``data/avatars/<id>/`` — only the small inference tensors (latents / face
crops) are shipped to the worker.

This mirrors the local backends exactly; keep the paste_back math in sync with
avatars/musetalk_avatar.py / avatars/wav2lip_avatar.py.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from avatars.base_avatar import BaseAvatar
from remote.client import RemoteClient, RemoteWorkerError
from utils.logger import logger
from registry import register

# The client the module-level load_model() created. Mirror of the local
# backends' module-global `device`. Only one model is ever resident.
_client: RemoteClient | None = None
_backend: str = "musetalk"
# Which avatar the WORKER currently has resident. The Mac caches avatars
# (global_avatars), so switching back to a cached avatar would otherwise leave
# the worker driving the wrong latents.
_worker_avatar_id: str = ""


# ─── module-level loaders (app.py contract) ─────────────────────────────────
def _backend_name() -> str:
    return _backend


def _as_numpy(tensor) -> np.ndarray:
    """torch.Tensor -> numpy (for the wire); numpy passes through."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def load_model(opt=None):
    global _client, _backend, _worker_avatar_id
    # Admin-UI address wins over --remote_worker / LT_REMOTE_WORKER.
    from remote.settings import effective
    address, source = effective()
    if not address:
        raise RuntimeError(
            "No remote worker address. Set it in the admin console "
            "(/admin.html -> Remote GPU Worker), pass --remote_worker host:port, "
            "or set LT_REMOTE_WORKER=host:port (e.g. 192.168.1.42:50051).")
    logger.info('[remote] using worker %s (source: %s)', address, source)
    _backend = (getattr(opt, "remote_backend", None)
                or os.environ.get("LT_REMOTE_BACKEND", "musetalk"))
    _worker_avatar_id = ""
    _client = RemoteClient(address)
    info = _client.wait_ready(deadline=30.0)
    logger.info('[remote] worker %s ready: device=%s gpu=%s',
                address, info.get("device"), info.get("detail"))
    try:
        _client.load_model(_backend)
    except RemoteWorkerError as exc:
        logger.error('[remote] LoadModel failed: %s', exc)
        raise
    logger.info('[remote] backend %r loaded on worker', _backend)
    return _client


def load_avatar(avatar_id: str):
    """Load paste-back artifacts locally and ship inference tensors to the worker.

    Returns the same tuple shape as the underlying backend so the avatar class
    can reuse its paste_back math.
    """
    if _client is None:
        raise RuntimeError("load_model() must run before load_avatar()")
    backend = _backend_name()

    if backend == "musetalk":
        from avatars.musetalk_avatar import load_avatar as _local
        frame_list, mask_list, coord_list, mask_coords, latents = _local(avatar_id)
        _ship_to_worker(backend, avatar_id, [_as_numpy(l) for l in latents])
        return frame_list, mask_list, coord_list, mask_coords, latents

    if backend == "wav2lip":
        from avatars.wav2lip_avatar import load_avatar as _local
        frame_list, face_list, coord_list = _local(avatar_id)
        _ship_to_worker(backend, avatar_id, list(face_list))
        return frame_list, face_list, coord_list

    raise ValueError(f"unknown remote_backend {backend!r}")


def _ship_to_worker(backend: str, avatar_id: str, frames) -> None:
    """Make `avatar_id` the resident avatar on the worker (idempotent)."""
    global _worker_avatar_id
    if _client is None:
        raise RuntimeError("remote client not initialised")
    try:
        _client.load_avatar(backend, avatar_id, frames)
    except RemoteWorkerError as exc:
        logger.error('[remote] LoadAvatar(%s) failed: %s', avatar_id, exc)
        raise
    _worker_avatar_id = avatar_id
    logger.info('[remote] avatar %r resident on worker', avatar_id)


@torch.no_grad()
def warm_up(batch_size, model):
    """Prime the worker's CUDA path with one dummy batch (matches local warmups)."""
    if _client is None:
        return
    backend = _backend_name()
    try:
        if backend == "musetalk":
            dummy = [np.ones((50, 384), dtype=np.uint8)] * batch_size
        else:
            dummy = [np.ones((1, 80, 16), dtype=np.float32)] * batch_size
        _client.infer_batch(0, batch_size, dummy)
        logger.info('[remote] worker warm-up ok')
    except Exception as exc:  # noqa: BLE001
        logger.warning('[remote] warm-up failed (first batch may be slow): %s', exc)


def unload_remote() -> None:
    """Tell the worker to free VRAM (called by app.unload_model)."""
    global _worker_avatar_id
    if _client is not None:
        try:
            _client.unload()
        except Exception as exc:  # noqa: BLE001
            logger.warning('[remote] worker unload failed: %s', exc)
    _worker_avatar_id = ""


# ─── avatar pipeline class ─────────────────────────────────────────────────
@register("avatar", "remote")
class RemoteAvatar(BaseAvatar):
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        self.client: RemoteClient = model
        self.backend = getattr(opt, "remote_backend", None) or _backend_name()
        self.avatar_id = opt.avatar_id

        if self.backend == "musetalk":
            (self.frame_list_cycle, self.mask_list_cycle, self.coord_list_cycle,
             self.mask_coords_list_cycle, self.input_latent_list_cycle) = avatar
            # The Mac caches avatars, so this avatar may be a cache hit while the
            # worker is driving a different one. Re-assert residency.
            if _worker_avatar_id != self.avatar_id:
                _ship_to_worker(self.backend, self.avatar_id,
                                [_as_numpy(l) for l in self.input_latent_list_cycle])
            from avatars.musetalk.whisper.audio2feature import Audio2Feature
            from avatars.audio_features.whisper import WhisperASR
            # Audio feature extraction stays on the Mac: whisper-tiny is small
            # and runs fine on MPS/CPU.
            self._audio_processor = Audio2Feature(model_path="./models/whisper")
            self.asr = WhisperASR(opt, self, self._audio_processor)
        elif self.backend == "wav2lip":
            self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar
            if _worker_avatar_id != self.avatar_id:
                _ship_to_worker(self.backend, self.avatar_id, list(self.face_list_cycle))
            from avatars.audio_features.mel import MelASR
            self.asr = MelASR(opt, self)
        else:
            raise ValueError(f"unknown remote backend {self.backend!r}")

        self.asr.warm_up()

    @torch.no_grad()
    def inference_batch(self, index, audiofeat_batch):
        # Raw pred crops come back from the worker; paste_back runs locally.
        return self.client.infer_batch(index, self.batch_size, audiofeat_batch)

    def paste_back_frame(self, pred_frame, idx: int):
        if self.backend == "musetalk":
            from avatars.musetalk.myutil import get_image_blending
            bbox = self.coord_list_cycle[idx]
            ori_frame = self.frame_list_cycle[idx].copy()
            x1, y1, x2, y2 = bbox
            res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            mask = self.mask_list_cycle[idx]
            mask_crop_box = self.mask_coords_list_cycle[idx]
            return get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

        if self.backend == "wav2lip":
            bbox = self.coord_list_cycle[idx]
            combine_frame = self.frame_list_cycle[idx].copy()
            y1, y2, x1, x2 = bbox
            res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            combine_frame[y1:y2, x1:x2] = res_frame
            return combine_frame

        raise ValueError(f"unknown remote backend {self.backend!r}")
