"""Client for the remote GPU worker (runs on the Mac, inside the app server).

Design notes
------------
* Synchronous gRPC stub. The avatar inference loop and the ASR worker are plain
  ``threading.Thread`` daemons (not the aiohttp event loop), so a blocking stub
  is the natural fit — it blocks exactly the thread that would otherwise be
  doing local GPU work.
* One channel per client, reused across batches (keepalive matters at 25 fps).
* Frames travel as uint8 BGR arrays; the Mac does paste_back locally, so only
  the small model crops cross the wire, not full frames.
"""
from __future__ import annotations

import os
import time
from typing import Iterable, Optional

import grpc
import numpy as np

from remote.proto import inference_pb2, inference_pb2_grpc
from remote.tensors import from_tensor, to_tensor
from utils.logger import logger

# 512 MB: one enhance job ships a whole clip in and a zip out.
_MAX_MSG = 512 * 1024 * 1024

_CHANNEL_OPTS = [
    ("grpc.max_send_message_length", _MAX_MSG),
    ("grpc.max_receive_message_length", _MAX_MSG),
    ("grpc.keepalive_time_ms", 20000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 1),
]


class RemoteWorkerError(RuntimeError):
    pass


class RemoteClient:
    """Thin wrapper over RemoteWorkerService."""

    def __init__(self, address: str, *, timeout: float = 30.0):
        self.address = address
        self.timeout = timeout
        self._channel = grpc.insecure_channel(address, options=_CHANNEL_OPTS)
        self._stub = inference_pb2_grpc.RemoteWorkerStub(self._channel)

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def from_env(cls) -> Optional["RemoteClient"]:
        addr = os.environ.get("LT_REMOTE_WORKER")
        if not addr:
            return None
        return cls(addr)

    def close(self) -> None:
        try:
            self._channel.close()
        except Exception:  # noqa: BLE001
            pass

    def health(self) -> dict:
        reply = self._stub.Health(inference_pb2.Empty(), timeout=self.timeout)
        return {"ok": reply.ok, "device": reply.device,
                "backend": reply.backend, "avatar_id": reply.avatar_id,
                "detail": reply.detail}

    def wait_ready(self, deadline: float = 30.0) -> dict:
        """Poll Health until the worker answers or the deadline passes."""
        end = time.time() + deadline
        last = None
        while time.time() < end:
            try:
                return self.health()
            except grpc.RpcError as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.5)
        raise RemoteWorkerError(f"worker {self.address} not reachable: {last}")

    # -- model / avatar ----------------------------------------------------
    def load_model(self, backend: str) -> None:
        reply = self._stub.LoadModel(
            inference_pb2.ModelSpec(backend=backend), timeout=self.timeout)
        if not reply.ok:
            raise RemoteWorkerError(f"LoadModel failed: {reply.msg}")

    def load_avatar(self, backend: str, avatar_id: str,
                    frames: Iterable[np.ndarray]) -> None:
        payload = inference_pb2.AvatarPayload(
            backend=backend, avatar_id=avatar_id,
            frames=[to_tensor(f) for f in frames])
        reply = self._stub.LoadAvatar(payload, timeout=self.timeout)
        if not reply.ok:
            raise RemoteWorkerError(f"LoadAvatar failed: {reply.msg}")

    def unload(self) -> None:
        self._stub.Unload(inference_pb2.Empty(), timeout=self.timeout)

    # -- inference ---------------------------------------------------------
    def infer_batch(self, index: int, batch_size: int,
                    audiofeat_batch: Iterable[np.ndarray]) -> np.ndarray:
        """Return the model's raw pred frames as a numpy array.

        Shape depends on backend: musetalk -> [B,H,W,3] uint8 BGR crops;
        wav2lip -> [B,H,W,3] float BGR. Callers paste_back locally.
        """
        request = inference_pb2.InferRequest(
            index=int(index), batch_size=int(batch_size),
            audiofeat=[to_tensor(a) for a in audiofeat_batch])
        reply = self._stub.InferBatch(request, timeout=self.timeout)
        return from_tensor(reply.frames)

    # -- enhance -----------------------------------------------------------
    def enhance_video(self, source_path: str, *, width: int, height: int,
                      background: str, fit: str, strength: float,
                      face_restore: bool, upscale: int,
                      max_frames: Optional[int], want_matte: bool,
                      want_person: bool,
                      timeout: float = 3600.0) -> inference_pb2.EnhanceReply:
        with open(source_path, "rb") as fh:
            source = fh.read()
        request = inference_pb2.EnhanceRequest(
            filename=os.path.basename(source_path), source=source,
            width=width, height=height, background=background, fit=fit,
            strength=strength, face_restore=face_restore, upscale=upscale,
            max_frames=max_frames or 0, want_matte=want_matte,
            want_person=want_person)
        reply = self._stub.EnhanceVideo(request, timeout=timeout)
        if not reply.ok:
            raise RemoteWorkerError(f"EnhanceVideo failed: {reply.msg}")
        return reply


def probe(address: str, timeout: float = 5.0) -> dict:
    """One-shot health probe (used by CLI --check and tests)."""
    client = RemoteClient(address, timeout=timeout)
    try:
        return client.health()
    finally:
        client.close()
