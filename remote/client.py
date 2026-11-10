"""Client for the remote GPU worker (runs on the host, inside the app server).

Design notes
------------
* Synchronous gRPC stub. The avatar inference loop and the ASR worker are plain
  ``threading.Thread`` daemons (not the aiohttp event loop), so a blocking stub
  is the natural fit — it blocks exactly the thread that would otherwise be
  doing local GPU work.
* One channel per client, reused across batches (keepalive matters at 25 fps).
* Frames travel as uint8 BGR arrays; the host does paste_back locally, so only
  the small model crops cross the wire, not full frames.
* ``PipelinedInfer`` runs two batches in flight so the ~0.6 MB/s link is not
  idle while the GPU computes. See its docstring for why that is worth it.
"""
from __future__ import annotations

import os
import threading
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

# Capabilities this client can use. The worker advertises what it implements in
# HealthReply.capabilities; anything missing here falls back to the old format,
# so host and worker can be updated in either order.
CAP_FEAT_BLOCK = "feat-block"
CAP_JPEG = "jpeg-bgr"



class RemoteWorkerError(RuntimeError):
    pass


class RemoteClient:
    """Thin wrapper over RemoteWorkerService."""

    def __init__(self, address: str, *, timeout: float = 30.0):
        self.address = address
        self.timeout = timeout
        self._channel = grpc.insecure_channel(address, options=_CHANNEL_OPTS)
        self._stub = inference_pb2_grpc.RemoteWorkerStub(self._channel)
        # Filled by health()/wait_ready(). Empty until the worker has been
        # asked, which is why every capability decision is "opt-in on evidence"
        # rather than a guess from a version number.
        self.capabilities: frozenset[str] = frozenset()

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def from_env(cls) -> Optional["RemoteClient"]:
        """Build a client for the effective worker address, or None.

        Goes through ``remote.settings.effective()`` -- NOT ``os.environ`` --
        because an address saved in the admin console legitimately wins over the
        env var, and reading the env directly would silently ignore it.
        """
        from remote.settings import effective_address
        address = effective_address()
        if not address:
            return None
        return cls(address)

    def close(self) -> None:
        try:
            self._channel.close()
        except Exception:  # noqa: BLE001
            pass

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def health(self) -> dict:
        reply = self._stub.Health(inference_pb2.Empty(), timeout=self.timeout)
        self.capabilities = frozenset(reply.capabilities)
        return {"ok": reply.ok, "device": reply.device,
                "backend": reply.backend, "avatar_id": reply.avatar_id,
                "detail": reply.detail, "revision": reply.revision,
                "busy": bool(reply.busy),
                "capabilities": sorted(self.capabilities)}

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
    def _build_request(self, index: int, batch_size: int, audiofeat_batch,
                       *, encoding: str = "jpeg-bgr", quality: int = 75,
                       feature_dtype: str = "float16") -> "inference_pb2.InferRequest":
        """Assemble an InferRequest, using the compact block when possible.

        The block path is only taken when the worker ADVERTISED ``feat-block``
        and the ASR stage attached a block. Otherwise we fall back to one tensor
        per frame, which every worker understands. That is what makes host and
        worker deployable in either order.
        """
        cast = np.float16 if feature_dtype == "float16" else np.float32
        block = getattr(audiofeat_batch, "block", None)
        offsets = getattr(audiofeat_batch, "offsets", ())
        rows = getattr(audiofeat_batch, "rows_per_frame", 0)
        request = inference_pb2.InferRequest(
            index=int(index), batch_size=int(batch_size),
            want_encoding=encoding, jpeg_quality=int(quality))
        if (block is not None and rows > 0 and offsets
                and self.supports(CAP_FEAT_BLOCK)):
            # A proto3 sub-message field is an immutable default until it is
            # touched; CopyFrom is the documented way to fill it in place.
            request.feat_block.CopyFrom(to_tensor(np.asarray(block, dtype=cast)))
            request.feat_offsets.extend(int(o) for o in offsets)
            request.feat_rows_per_frame = int(rows)
            return request
        if block is not None and not self.supports(CAP_FEAT_BLOCK) \
                and not getattr(self, "_no_block_logged", False):
            logger.warning(
                "[remote] worker at %s does not advertise %r (revision %s); "
                "sending per-frame audio features -- the upload is ~3x larger "
                "and fps stays link-bound. Redeploy the worker to fix.",
                self.address, CAP_FEAT_BLOCK, "unknown")
            self._no_block_logged = True
        request.audiofeat.extend(to_tensor(np.asarray(a, dtype=cast))
                                 for a in audiofeat_batch)
        return request

    def infer_batch(self, index: int, batch_size: int,
                    audiofeat_batch: Iterable[np.ndarray], *,
                    encoding: str = "jpeg-bgr",
                    quality: int = 75,
                    feature_dtype: str = "float16") -> np.ndarray:
        """Return the model's raw pred frames as a numpy array.

        Shape depends on backend: musetalk -> [B,H,W,3] uint8 BGR crops;
        wav2lip -> [B,H,W,3] float BGR. Callers paste_back locally.

        Three wire optimisations, all aimed at the slow SSH link rather than at
        GPU time (the GPU needs ~40 ms/batch and was never the bottleneck):

        * the COMPACT FEATURE BLOCK (see InferRequest.feat_block): consecutive
          frames of a batch read overlapping rows of one whisper array, so a
          batch of 8 ships 24 rows instead of 80 -- 150 KB instead of 600 KB
          per batch. Lossless: the host proves the reconstruction before
          sending. Requires a worker advertising ``feat-block``.
        * ``encoding="jpeg-bgr"`` (default) has the worker JPEG-compress the
          reply. Raw crops are ~192 KB/frame -> 1.57 MB/batch; q75 is ~35x
          smaller and the host re-encodes to VP8 anyway. Pass ``""`` for raw
          (tests/debugging).
        * ``feature_dtype="float16"`` (default) halves the UPLOAD. whisper
          features are float32 in memory, but the worker immediately casts them
          to the model dtype (fp16 on CUDA), so sending fp16 loses nothing.
          ``"float32"`` restores the old wire format.
        """
        request = self._build_request(index, batch_size, audiofeat_batch,
                                      encoding=encoding, quality=quality,
                                      feature_dtype=feature_dtype)
        reply = self._stub.InferBatch(request, timeout=self.timeout)
        return _decode_reply(reply)


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

    # -- avatar creation ---------------------------------------------------
    def generate_avatar(self, backend: str, avatar_id: str, *,
                        source_path: Optional[str] = None,
                        source_bytes: Optional[bytes] = None,
                        filename: str = "",
                        params: Optional[dict] = None,
                        timeout: float = 7200.0) -> bytes:
        """Build an avatar on the worker GPU; return the artifact zip bytes.

        ``source_path`` is a path the WORKER can read (only meaningful when the
        two machines share data/); ``source_bytes`` is for an uploaded clip. A
        generation runs per-frame VAE + parsing work, so the default timeout is
        two hours rather than the 30 s used for control RPCs.
        """
        import json as _json
        if not source_bytes and not source_path:
            raise ValueError("source_path or source_bytes is required")
        request = inference_pb2.GenerateAvatarRequest(
            backend=backend, avatar_id=avatar_id,
            source=source_bytes or b"", filename=filename,
            source_path=source_path or "",
            params_json=_json.dumps(params or {}))
        reply = self._stub.GenerateAvatar(request, timeout=timeout)
        if not reply.ok:
            raise RemoteWorkerError(f"GenerateAvatar failed: {reply.msg}")
        return reply.archive


def _decode_reply(reply) -> np.ndarray:
    """Turn an InferReply into the pred-frame array (jpeg or raw tensor)."""
    if reply.encoding == "jpeg-bgr":
        import cv2
        frames = []
        for blob in reply.jpeg:
            frame = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is None:
                raise RemoteWorkerError("worker sent undecodable jpeg frame")
            frames.append(frame)
        return np.stack(frames)
    return from_tensor(reply.frames)


class PipelinedInfer:
    """Run two InferBatch calls concurrently, preserving submission order.

    Why
    ---
    A single unary InferBatch leaves the link idle for the whole time the worker
    is doing GPU work and JPEG-encoding, and leaves the GPU idle while the reply
    is on the wire. Measured on this link: raw pipe 1.14 MB/s but the RPC path
    only sustains 0.59 MB/s -- the missing half is that round-trip stall. With
    two requests in flight the transfer of batch N+1 overlaps the compute of
    batch N, so throughput approaches whichever is slower instead of their sum.

    Why order matters
    -----------------
    Frames carry `index`, and `paste_back_frame` picks the avatar frame by
    `mirror_index(length, index)`; the render loop also advances its own index
    in lockstep. Returning frames out of order would therefore not merely
    reorder pixels, it would attach the wrong source frame to them. So results
    are re-ordered back into submission order here, and a failed batch poisons
    the slot rather than silently shifting later frames up.

    Concurrency safety
    ------------------
    The worker serialises GPU work behind one mutex anyway, so the second
    in-flight request overlaps TRANSPORT, not compute. That is exactly the
    stall being recovered, and it needs no worker change. Two gRPC stubs are
    used because a channel multiplexes concurrent RPCs, but the blocking stub
    would otherwise serialise on its own call.

    Shutdown
    --------
    ``close()`` must be called (or use it as a context manager) so the worker
    threads stop; a daemon thread would keep reading a closed channel.
    """

    def __init__(self, client: "RemoteClient", depth: int = 2):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.client = client
        self.depth = depth
        self._slots = {}                  # seq -> Future-like
        self._cond = threading.Condition()
        self._next_seq = 0
        self._next_out = 0
        self._inflight = 0
        self._closed = False

    # -- submit/collect ----------------------------------------------------
    def submit(self, index: int, batch_size: int, audiofeat_batch, **kw) -> int:
        """Queue one batch. Blocks while `depth` requests are already in flight."""
        with self._cond:
            while self._inflight >= self.depth and not self._closed:
                self._cond.wait(timeout=0.5)
            if self._closed:
                raise RemoteWorkerError("PipelinedInfer is closed")
            seq = self._next_seq
            self._next_seq += 1
            self._inflight += 1
            slot = {"done": False, "frames": None, "error": None}
            self._slots[seq] = slot
            request = self.client._build_request(index, batch_size, audiofeat_batch, **kw)

        def _run():
            frames, error = None, None
            try:
                frames = _decode_reply(
                    self.client._stub.InferBatch(request, timeout=self.client.timeout))
            except Exception as exc:  # noqa: BLE001
                error = exc
            with self._cond:
                slot["frames"] = frames
                slot["error"] = error
                slot["done"] = True
                self._inflight -= 1
                if error is not None:
                    logger.error("[remote] pipelined batch failed: %s", error)
                self._cond.notify_all()

        threading.Thread(target=_run, name=f"infer-rpc-{seq}", daemon=True).start()
        return seq

    def result(self, seq: int):
        """Block for batch `seq` and re-order it behind any earlier batch."""
        with self._cond:
            while True:
                if self._closed:
                    raise RemoteWorkerError("PipelinedInfer is closed")
                if seq < self._next_out:
                    raise RemoteWorkerError(f"batch {seq} was already consumed")
                # Release only in submission order: wait until MY batch is done
                # AND nothing earlier is still outstanding, so the caller's
                # index stays in lockstep with what it pastes back.
                ready = all(self._slots[s]["done"] for s in self._slots
                            if s <= seq)
                if ready and seq == self._next_out:
                    slot = self._slots.pop(seq)
                    self._next_out += 1
                    if slot["error"] is not None:
                        raise RemoteWorkerError(str(slot["error"]))
                    return slot["frames"]
                self._cond.wait(timeout=0.5)

    def run(self, index: int, batch_size: int, audiofeat_batch, **kw):
        """submit() + result() for the simple case."""
        return self.result(self.submit(index, batch_size, audiofeat_batch, **kw))

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def probe(address: str, timeout: float = 5.0) -> dict:
    """One-shot health probe (used by CLI --check and tests)."""
    client = RemoteClient(address, timeout=timeout)
    try:
        return client.health()
    finally:
        client.close()
