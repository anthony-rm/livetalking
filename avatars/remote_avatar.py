"""Remote-GPU avatar backend (Option B).

Registers ``@register("avatar", "remote")``. Use with::

    ./run.sh --worker_mode remote --remote_worker 192.168.1.42:50051 \\
             --remote_backend musetalk --avatar_id my_avatar

(or flip the toggle in /admin.html -> Remote GPU Worker).

The local host keeps the whole pipeline (TTS, ASR feature extraction, paste-back,
streaming). Only ``inference_batch`` is offloaded: audio features go to the
remote GPU worker, pred crops come back, paste_back runs locally. The model
uses the avatar artifact metadata (frame list, masks, coords) from the local
``data/avatars/<id>/`` — only the small inference tensors (latents / face
crops) are shipped to the worker.

This mirrors the local backends exactly; keep the paste_back math in sync with
avatars/musetalk_avatar.py / avatars/wav2lip_avatar.py.
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np
import torch

from avatars.base_avatar import BaseAvatar
from avatars import detect as avatar_detect
from remote.client import RemoteClient, RemoteWorkerError
from utils.logger import logger
from registry import register

# Backends the WORKER can run. ultralight exists as a local host backend only.
WORKER_BACKENDS = ("musetalk", "wav2lip")

# The client the module-level load_model() created. Mirror of the local
# backends' module-global `device`. Only one model is ever resident.
_client: RemoteClient | None = None
_backend: str = ""
# Configured --remote_backend, used only as a last-resort fallback when no
# avatar is loaded (and thus no artifact has declared the real backend).
_fallback_backend: str = "musetalk"
# Which avatar the WORKER currently has resident. The host caches avatars
# (global_avatars), so switching back to a cached avatar would otherwise leave
# the worker driving the wrong latents.
_worker_avatar_id: str = ""
# avatar_id -> backend its cached payload was loaded with. The host caches
# avatars across sessions (global_avatars), and a cached musetalk payload is a
# 5-tuple while a wav2lip one is a 3-tuple, so the class must know which it got
# instead of assuming the startup backend.
_avatar_backends: dict[str, str] = {}
# Serialises backend switching + avatar shipping: two concurrent connects for
# avatars of different backends would otherwise interleave LoadModel/LoadAvatar
# calls and leave the worker holding one backend while the other session thinks
# it loaded another.
_load_lock = threading.Lock()


# ─── module-level loaders (app.py contract) ─────────────────────────────────
def _backend_name() -> str:
    return _backend


def _ensure_worker_backend(backend: str) -> None:
    """Switch the worker to `backend`, loading its weights if needed.

    The worker holds ONE backend at a time by design, so driving a wav2lip
    avatar after a musetalk one (or the reverse) means the weights are swapped
    and the resident avatar dropped. That is a slow call the first time for a
    given backend; log it so a slow connect is explained rather than mysterious.

    ``_backend`` is '' until the worker actually holds weights: load_model()
    only connects, deferring the (heavy) weight load to the first avatar, whose
    artifacts decide which backend the worker needs -- loading the configured
    --remote_backend up front just to swap it a moment later wastes a reload.
    """
    global _backend, _worker_avatar_id
    if backend not in WORKER_BACKENDS:
        raise RuntimeError(
            f"the remote worker cannot drive {backend!r}; it supports "
            f"{list(WORKER_BACKENDS)}. Run that avatar on the host instead "
            f"(--model {backend}) or regenerate it with a supported model.")
    if _client is None:
        raise RuntimeError("load_model() must run before load_avatar()")
    if _backend == backend:
        return
    if _backend:
        # Swapping the worker's backend invalidates it for any live session:
        # inference_batch would run the old backend's features through the new
        # weights. Refuse, mirroring unload_model's stance.
        from server.session_manager import session_manager
        active = session_manager.active_sessions()
        if active:
            raise RuntimeError(
                f"cannot switch the remote worker from {_backend!r} to "
                f"{backend!r}: {active} session(s) are still connected. "
                f"Disconnect them first, then connect the {backend!r} avatar.")
        logger.info('[remote] switching worker backend %s -> %s (reloads weights)',
                    _backend, backend)
    else:
        logger.info('[remote] loading backend %r on the worker', backend)
    try:
        _client.load_model(backend)
    except RemoteWorkerError as exc:
        logger.error('[remote] loading backend %r failed: %s', backend, exc)
        raise
    _backend = backend
    # Different backend => whatever the worker had resident is gone.
    _worker_avatar_id = ""



def _as_numpy(tensor) -> np.ndarray:
    """torch.Tensor -> numpy (for the wire); numpy passes through."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def load_model(opt=None):
    """Connect to the worker. Does NOT load weights.

    Which backend's weights the worker needs is decided by the first avatar's
    artifacts (see _ensure_worker_backend), so priming one here would just be
    reloaded a moment later whenever the avatar belongs to the other backend.
    """
    global _client, _backend, _worker_avatar_id, _fallback_backend
    # Admin-UI address wins over --remote_worker / LT_REMOTE_WORKER.
    from remote.settings import effective
    address, source = effective()
    if not address:
        raise RuntimeError(
            "No remote worker address. Set it in the admin console "
            "(/admin.html -> Remote GPU Worker), pass --remote_worker host:port, "
            "or set LT_REMOTE_WORKER=host:port (e.g. 192.168.1.42:50051).")
    logger.info('[remote] using worker %s (source: %s)', address, source)
    # Nothing is resident on the worker yet as far as we know. The configured
    # --remote_backend is only a fallback for callers that never load an avatar.
    _backend = ""
    _fallback_backend = (getattr(opt, "remote_backend", None)
                         or os.environ.get("LT_REMOTE_BACKEND", "musetalk"))
    _worker_avatar_id = ""
    _client = RemoteClient(address)
    info = _client.wait_ready(deadline=30.0)
    logger.info('[remote] worker %s ready: device=%s gpu=%s (backend %r, '
                'loaded on first avatar)', address, info.get("device"),
                info.get("detail"), _fallback_backend)
    return _client


def load_avatar(avatar_id: str, backend: str | None = None):
    """Load paste-back artifacts locally and ship inference tensors to the worker.

    The backend is derived from the avatar's own artifacts unless the caller
    passes one explicitly. Detecting it (rather than trusting the startup
    --remote_backend) is what lets one server serve a mixed ``data/avatars``
    directory: connecting a wav2lip avatar while the worker was primed for
    musetalk switches the worker over instead of crashing on a missing
    latents.pt.

    Returns the same tuple shape as the underlying backend so the avatar class
    can reuse its paste_back math.
    """
    if _client is None:
        raise RuntimeError("load_model() must run before load_avatar()")

    if backend is None:
        detected, reason = avatar_detect.detect_backend(avatar_id)
        if detected is None:
            raise RuntimeError(
                f"cannot determine a backend for avatar '{avatar_id}': {reason}")
        backend = detected
        logger.info('[remote] avatar %r -> backend %r (%s)',
                    avatar_id, backend, reason)

    with _load_lock:
        _ensure_worker_backend(backend)
        _avatar_backends[avatar_id] = backend

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
    backend = _backend or _fallback_backend
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
    global _worker_avatar_id, _backend
    if _client is not None:
        try:
            _client.unload()
        except Exception as exc:  # noqa: BLE001
            logger.warning('[remote] worker unload failed: %s', exc)
    # The worker's Unload drops its backend entirely, so forget it here too --
    # otherwise the next connect would skip the LoadModel it now needs.
    _worker_avatar_id = ""
    _backend = ""
    _avatar_backends.clear()

# ─── avatar pipeline class ─────────────────────────────────────────────────
@register("avatar", "remote")
class RemoteAvatar(BaseAvatar):
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        self.client: RemoteClient = model
        self.avatar_id = opt.avatar_id

        # The payload's shape is determined by the backend its artifacts belong
        # to, NOT by the startup --remote_backend. A cached avatar may have been
        # loaded under a different backend than the one this session started
        # with, so look it up instead of trusting opt.
        detected, reason = avatar_detect.detect_backend(self.avatar_id)
        self.backend = _avatar_backends.get(self.avatar_id) or detected \
            or getattr(opt, "remote_backend", None) or _backend_name()

        if self.backend == "musetalk":
            (self.frame_list_cycle, self.mask_list_cycle, self.coord_list_cycle,
             self.mask_coords_list_cycle, self.input_latent_list_cycle) = avatar
            # The host caches avatars, so this avatar may be a cache hit while the
            # worker is driving a different one. Re-assert residency (and the
            # backend, which switches weights if the worker moved on).
            with _load_lock:
                _ensure_worker_backend(self.backend)
                if _worker_avatar_id != self.avatar_id:
                    _ship_to_worker(self.backend, self.avatar_id,
                                    [_as_numpy(l) for l in self.input_latent_list_cycle])
            from avatars.musetalk.whisper.audio2feature import Audio2Feature
            from avatars.audio_features.whisper import WhisperASR
            # Audio feature extraction stays on the host: whisper-tiny is small
            # and runs fine on MPS/CPU.
            self._audio_processor = Audio2Feature(model_path="./models/whisper")
            self.asr = WhisperASR(opt, self, self._audio_processor)
        elif self.backend == "wav2lip":
            self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar
            with _load_lock:
                _ensure_worker_backend(self.backend)
                if _worker_avatar_id != self.avatar_id:
                    _ship_to_worker(self.backend, self.avatar_id, list(self.face_list_cycle))
            from avatars.audio_features.mel import MelASR
            self.asr = MelASR(opt, self)
        else:
            raise ValueError(
                f"unknown remote backend {self.backend!r} for avatar "
                f"{self.avatar_id!r} ({reason})")

        self.asr.warm_up()

        # Overlap transport with compute: inference is a network round-trip
        # here, so a second batch in flight recovers the link idle time (see
        # remote.client.PipelinedInfer). Opt-in, and only when the opt asks for
        # it, so a debugging run can stay strictly synchronous.
        self.infer_pipeline = None
        depth = int(getattr(opt, "remote_pipeline", 2) or 0)
        if depth > 0:
            from remote.client import PipelinedInfer
            self.infer_pipeline = PipelinedInfer(self.client, depth=depth)
            logger.info("[remote] pipelining inference: %d batch(es) in flight", depth)

    @torch.no_grad()
    def inference_batch(self, index, audiofeat_batch):
        # Raw pred crops come back from the worker; paste_back runs locally.
        # This is the SYNCHRONOUS path, kept for direct calls and for
        # remote_pipeline=0; the render loop goes through the pipeline instead.
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
