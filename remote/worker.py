"""Remote GPU worker — runs on the Linux/NVIDIA box.

Start it on the GPU server (same repo checkout, conda env with CUDA torch):

    python -m remote.worker --backend musetalk

It binds 127.0.0.1:50051 by default: the protocol is plaintext and
unauthenticated, so it is reachable ONLY through an SSH local-forward (the host
opens one automatically, see remote/tunnel.py). Pass --host 0.0.0.0 only on a
trusted LAN where you accept the exposure.

The host server talks to it via remote/client.py; nothing else here is exposed.
This process owns ALL GPU work: avatar inference (UNet+VAE / Wav2Lip) and the
base-video enhancer (RVM matting + Real-ESRGAN + GFPGAN). The host keeps TTS,
ASR feature extraction, paste-back, and streaming.

Only ONE backend + ONE avatar is resident at a time (the app is single-model by
design). Loading a different backend/avatar replaces the resident one.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import zipfile
from concurrent import futures
from typing import Optional

import grpc
import numpy as np

from remote.proto import inference_pb2, inference_pb2_grpc
from remote.tensors import from_tensor, to_tensor
from utils.image import mirror_index
from utils.logger import logger

_MAX_MSG = 512 * 1024 * 1024

# Feature width the musetalk UNet consumes (pe d_model=384). A frame's chunk is
# (rows * hidden_states) x 384, e.g. (50, 384) for the 10-row/5-state window.
# The compact feature block arrives as (span, hidden_states, 384), so rebuilding
# a frame means flattening every leading dim into the row axis and keeping this
# as the column count -- NOT block.shape[-1] scaled by anything.
FEAT_DIM = 384


# Cached per-process revision of the code this worker was started from.
_REVISION: Optional[str] = None


def _revision() -> str:
    """Code revision this worker process is serving (see remote/revision.py).

    Computed once per process: the value describes the code that was loaded at
    start-up, so recomputing it after a deploy swapped files on disk would
    report the NEW revision while the OLD code is still resident -- exactly the
    lie this field exists to prevent. Restart the worker to pick up a new one.
    """
    global _REVISION
    if _REVISION is None:
        from remote.revision import current_revision
        _REVISION = current_revision()
        logger.info(f"[worker] serving revision {_REVISION or 'unknown'}")
    return _REVISION


# ─── Inference backends ────────────────────────────────────────────────────
# Each backend exposes: load(), avatar_artifacts(avatar_id), infer(index, batch,
# feats) -> np.ndarray. The inference math mirrors the local avatar modules
# (avatars/musetalk_avatar.py, avatars/wav2lip_avatar.py) exactly; keep them in
# sync if the model code changes.
class MuseTalkBackend:
    name = "musetalk"

    def __init__(self) -> None:
        self.model = None       # (vae, unet, pe, timesteps, audio_processor)
        self.latents = None     # list of [1,8,32,32] tensors
        import torch
        from utils.device import initialize_device
        self.torch = torch
        self.device = initialize_device()
        logger.info(f"[worker] musetalk device={self.device}")

    def load_model(self) -> None:
        """Load UNet + VAE + positional encoding, WITHOUT the audio processor.

        ``avatars.musetalk_avatar.load_model()`` also builds an Audio2Feature
        (whisper-tiny: ~144 MB of weights and VRAM) because the LOCAL pipeline
        needs it. On the worker the audio features arrive pre-computed over the
        wire, so loading whisper is pure waste -- and it made ./models/whisper a
        deployment requirement for no benefit.
        """
        from avatars.musetalk.models.vae import VAE
        from avatars.musetalk.models.unet import UNet, PositionalEncoding

        vae = VAE(model_path=os.path.join("models", "sd-vae"))
        unet = UNet(unet_config=os.path.join("models", "musetalkV15", "musetalk.json"),
                    model_path=os.path.join("models", "musetalkV15", "unet.pth"),
                    device=self.device)
        pe = PositionalEncoding(d_model=384)

        timesteps = self.torch.tensor([0], device=self.device)
        # fp16 only where it is actually accelerated; emulated fp16 on CPU is
        # slower than fp32 (same rule the local backends follow).
        if self.device.type in ("cuda", "mps"):
            pe = pe.half().to(self.device)
            vae.vae = vae.vae.half().to(self.device)
            unet.model = unet.model.half().to(self.device)
        self.model = (vae, unet, pe, timesteps, None)
        logger.info("[worker] musetalk weights loaded (unet+pe+vae, no whisper)")

    def avatar_artifacts(self, avatar_id: str):
        """Inference-only artifacts (latents). paste_back metadata stays on the host."""
        import torch
        path = f"./data/avatars/{avatar_id}/latents.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return torch.load(path, map_location=self.device)

    def load_avatar(self, avatar_id: str, frames) -> None:
        # Prefer the on-disk latents if this worker has the avatar; otherwise use
        # the tensors shipped over the wire (keeps both machines independent).
        try:
            self.latents = self.avatar_artifacts(avatar_id)
        except Exception:  # noqa: BLE001
            self.latents = [self.torch.from_numpy(from_tensor(t)) for t in frames]
        logger.info(f"[worker] musetalk avatar {avatar_id!r} resident "
                    f"({len(self.latents)} frames)")

    def infer(self, index: int, batch_size: int, audiofeat: np.ndarray) -> np.ndarray:
        torch = self.torch
        vae, unet, pe, timesteps, _ = self.model
        length = len(self.latents)
        whisper_batch = np.stack(audiofeat)
        latent_batch = [self.latents[mirror_index(length, index + i)]
                        for i in range(batch_size)]
        latent_batch = torch.cat(latent_batch, dim=0)
        with torch.no_grad():
            # .to(dtype=...) below makes the WIRE dtype irrelevant: the host may
            # send fp16 (half the bytes over a slow link) or fp32 and both land
            # as the model's dtype here. Keep the cast, do not assume fp32.
            audio_feature_batch = torch.from_numpy(whisper_batch).to(
                device=unet.device, dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(device=unet.device, dtype=unet.model.dtype)
            pred_latents = unet.model(
                latent_batch, timesteps,
                encoder_hidden_states=audio_feature_batch).sample
            pred = vae.decode_latents(pred_latents)
        return pred


class Wav2LipBackend:
    name = "wav2lip"

    def __init__(self) -> None:
        self.model = None
        self.faces = None
        import torch
        from utils.device import initialize_device
        self.torch = torch
        self.device = initialize_device()
        logger.info(f"[worker] wav2lip device={self.device}")

    def load_model(self) -> None:
        from avatars.wav2lip_avatar import load_model
        self.model = load_model("./models/wav2lip.pth")
        logger.info("[worker] wav2lip weights loaded")

    def avatar_artifacts(self, avatar_id: str):
        from avatars.wav2lip_avatar import load_avatar
        # returns (frame_list_cycle, face_list_cycle, coord_list_cycle)
        _, face_list, _ = load_avatar(avatar_id)
        return face_list

    def load_avatar(self, avatar_id: str, frames) -> None:
        try:
            self.faces = self.avatar_artifacts(avatar_id)
        except Exception:  # noqa: BLE001
            self.faces = [from_tensor(t) for t in frames]
        logger.info(f"[worker] wav2lip avatar {avatar_id!r} resident "
                    f"({len(self.faces)} faces)")

    def infer(self, index: int, batch_size: int, audiofeat: np.ndarray) -> np.ndarray:
        torch = self.torch
        length = len(self.faces)
        img_batch = [self.faces[mirror_index(length, index + i)] for i in range(batch_size)]
        img_batch = np.asarray(img_batch)
        audiofeat = np.asarray(audiofeat)
        face = img_batch[0]
        img_masked = img_batch.copy()
        img_masked[:, face.shape[0] // 2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        audiofeat = np.reshape(audiofeat, [len(audiofeat), audiofeat.shape[1],
                                           audiofeat.shape[2], 1])
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
        audiofeat = torch.FloatTensor(np.transpose(audiofeat, (0, 3, 1, 2))).to(self.device)
        fp16 = next(self.model.parameters()).dtype == torch.float16
        if fp16:
            img_batch = img_batch.half()
            audiofeat = audiofeat.half()
        with torch.no_grad():
            pred = self.model(audiofeat, img_batch)
        return (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.float32)


_BACKENDS = {"musetalk": MuseTalkBackend, "wav2lip": Wav2LipBackend}


# ─── Servicer ──────────────────────────────────────────────────────────────
class RemoteWorkerServicer(inference_pb2_grpc.RemoteWorkerServicer):
    def __init__(self, default_backend: Optional[str] = None) -> None:
        self.backend = None
        self.backend_name = ""
        self.avatar_id = ""
        # A single mutex serialises GPU work: inference, model load, and
        # enhance never overlap (VRAM + the enhance stack is not reentrant).
        import threading
        self.lock = threading.Lock()
        # Distinguishes "idle" from "a GPU call is in flight". The mutex alone
        # cannot, because Health must not block behind a 3-minute enhance job.
        self._in_flight = threading.Event()
        if default_backend:
            self._ensure_backend(default_backend)
        try:
            import torch
            self.device_name = str(torch.device("cuda")) if torch.cuda.is_available() \
                else "cpu"
        except Exception:  # noqa: BLE001
            self.device_name = "unknown"

    def _ensure_backend(self, name: str):
        if self.backend is not None and self.backend_name == name:
            return
        cls = _BACKENDS.get(name)
        if cls is None:
            raise ValueError(f"unknown backend {name!r}; have {list(_BACKENDS)}")
        self.backend = cls()
        self.backend.name = name
        self.backend_name = name
        self.avatar_id = ""

    # -- RPCs --------------------------------------------------------------
    # Wire features this worker implements (see HealthReply.capabilities).
    # The host falls back to the legacy per-frame `audiofeat` tensors and raw
    # reply frames when a name is absent, so advertising these is what lets a
    # new host and an old worker (or the reverse) still work together.
    CAPABILITIES = ("feat-block", "jpeg-bgr")

    @contextlib.contextmanager
    def _gpu_work(self):
        """Hold the GPU mutex and mark the worker busy for the duration.

        ``Health`` reports ``busy`` so a deploy can refuse to restart the worker
        while a session is mid-batch instead of killing it silently.
        """
        with self.lock:
            self._in_flight.set()
            try:
                yield
            finally:
                self._in_flight.clear()

    def Health(self, request, context):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu = ""
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        return inference_pb2.HealthReply(
            ok=True, device=device, backend=self.backend_name,
            avatar_id=self.avatar_id, detail=gpu,
            revision=_revision(), busy=self._in_flight.is_set(),
            capabilities=list(self.CAPABILITIES))

    def LoadModel(self, request, context):
        with self._gpu_work():
            try:
                self._ensure_backend(request.backend)
                if self.backend.model is None:
                    self.backend.load_model()
                    self.backend_name = request.backend
                return inference_pb2.OpReply(ok=True, msg=f"loaded {request.backend}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("[worker] LoadModel failed")
                return inference_pb2.OpReply(ok=False, msg=str(exc))

    def LoadAvatar(self, request, context):
        with self._gpu_work():
            try:
                self._ensure_backend(request.backend)
                if self.backend.model is None:
                    self.backend.load_model()
                self.backend.load_avatar(request.avatar_id, request.frames)
                self.avatar_id = request.avatar_id
                return inference_pb2.OpReply(ok=True, msg=f"avatar {request.avatar_id}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("[worker] LoadAvatar failed")
                return inference_pb2.OpReply(ok=False, msg=str(exc))

    def _request_features(self, request):
        """Per-frame audio features, from the compact block or legacy tensors.

        The block form is what makes the upload 3.3x smaller: the host sends the
        span of whisper rows the whole batch reads plus each frame's offset, and
        we rebuild the per-frame windows here. The host PROVED that
        ``block[off:off+rows]`` equals the tensor it used to send, so the reshape
        below is a reconstruction, not an approximation.

        Rejecting a malformed block is deliberate: silently falling back would
        let the host believe it saved 3x the bytes while the frames are wrong.
        """
        if request.feat_block.data and request.feat_offsets:
            rows = int(request.feat_rows_per_frame)
            if rows <= 0:
                raise ValueError("feat_block sent without feat_rows_per_frame")
            block = from_tensor(request.feat_block)
            feats = []
            for frame_i, offset in enumerate(request.feat_offsets):
                lo, hi = int(offset), int(offset) + rows
                if lo < 0 or hi > block.shape[0]:
                    raise ValueError(
                        f"feat_block[{frame_i}] window [{lo},{hi}) is outside the "
                        f"block's {block.shape[0]} rows")
                window = block[lo:hi]
                if window.size % FEAT_DIM:
                    raise ValueError(
                        f"feat_block[{frame_i}] window of {window.size} values is "
                        f"not a multiple of the {FEAT_DIM}-wide feature")
                # Collapse the leading dims into rows: (rows, H, 384) -> (rows*H, 384)
                feats.append(window.reshape(-1, FEAT_DIM))
            expected = int(request.batch_size)
            if expected and len(feats) != expected:
                raise ValueError(
                    f"feat_block yielded {len(feats)} frames, batch_size is "
                    f"{expected}")
            return feats
        return [from_tensor(t) for t in request.audiofeat]

    def InferBatch(self, request, context):
        with self._gpu_work():
            try:
                if self.backend is None or self.backend.model is None:
                    raise RuntimeError("no model loaded; call LoadModel first")
                feats = self._request_features(request)
                pred = self.backend.infer(int(request.index),
                                          int(request.batch_size), feats)
                # Compress on the responder's side of the wire. The pred crops are
                # full-resolution uint8 images, so JPEG is the right codec (25x at
                # q90 for ~1.3/255 mean error); it costs ~6 ms of CPU per batch
                # and saves ~2 s of a ~1 MB/s SSH link.
                if request.want_encoding == "jpeg-bgr":
                    try:
                        import numpy as np
                        import cv2
                        quality = int(request.jpeg_quality) or 90
                        jpegs = []
                        for frame in pred:
                            # musetalk returns uint8 BGR crops; wav2lip returns
                            # float32 already in 0..255. JPEG needs uint8, and
                            # clipping keeps imencode from wrapping a >255 value.
                            arr = np.asarray(frame)
                            if arr.dtype != np.uint8:
                                arr = np.clip(arr, 0, 255).astype(np.uint8)
                            ok, buf = cv2.imencode(
                                ".jpg", arr,
                                [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                            if not ok:
                                raise RuntimeError("cv2.imencode returned False")
                            jpegs.append(buf.tobytes())
                        return inference_pb2.InferReply(
                            jpeg=jpegs, encoding="jpeg-bgr")
                    except Exception:  # noqa: BLE001
                        # Never fail a frame because compression hiccuped: log and
                        # fall through to the raw reply.
                        logger.exception("[worker] jpeg encode failed; sending raw")
                return inference_pb2.InferReply(frames=to_tensor(pred))
            except Exception as exc:  # noqa: BLE001
                logger.exception("[worker] InferBatch failed")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return inference_pb2.InferReply()

    def Unload(self, request, context):
        with self._gpu_work():
            self.backend = None
            self.backend_name = ""
            self.avatar_id = ""
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # drop the enhancer stack too
                from tools.enhance import pipeline
                pipeline.reset_enhancer()
                pipeline.reset_matte_state()
            except Exception:  # noqa: BLE001
                pass
            return inference_pb2.OpReply(ok=True, msg="unloaded")

    def EnhanceVideo(self, request, context):
        # Enhance is a batch job; it does not need the inference lock held for
        # its whole duration but it DOES share VRAM, so serialise it against
        # inference by taking the same lock for the pipeline setup only is not
        # enough — run it fully serialised to avoid OOM thrash.
        with self._gpu_work():
            try:
                import json
                from tools.enhance import pipeline

                work = tempfile.mkdtemp(prefix="lt_enhance_")
                ext = os.path.splitext(request.filename or "")[1].lower() or ".mp4"
                src = os.path.join(work, f"source{ext}")
                with open(src, "wb") as fh:
                    fh.write(request.source)
                out = os.path.join(work, "enhanced.mp4")
                matte_dir = os.path.join(work, "matte") if request.want_matte else None
                person_dir = os.path.join(work, "person") if request.want_person else None

                result = pipeline.enhance_video(
                    src, out,
                    width=request.width, height=request.height,
                    background=request.background, fit=request.fit,
                    strength=request.strength,
                    face_restore=request.face_restore,
                    upscale=request.upscale,
                    max_frames=(request.max_frames or None),
                    matte_out=matte_dir, person_out=person_dir)

                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(out, "enhanced.mp4")
                    for sub, name in ((matte_dir, "matte"), (person_dir, "person")):
                        if sub and os.path.isdir(sub):
                            for root, _dirs, files in os.walk(sub):
                                for f in files:
                                    full = os.path.join(root, f)
                                    rel = os.path.relpath(full, sub)
                                    zf.write(full, f"{name}/{rel}")
                    zf.writestr("meta.json", json.dumps(result, indent=2))
                return inference_pb2.EnhanceReply(
                    ok=True, msg="ok", archive=buf.getvalue(),
                    frames=result.get("frames", 0), fps=result.get("fps", 0.0))
            except Exception as exc:  # noqa: BLE001
                logger.exception("[worker] EnhanceVideo failed")
                return inference_pb2.EnhanceReply(ok=False, msg=str(exc))

    def GenerateAvatar(self, request, context):
        """Build avatar artifacts on the worker GPU and return them as a zip.

        Runs the SAME genavatar code the host would run locally, so the artifact
        layout is identical and the host never needs a box-specific path. The
        GPU mutex is held for the whole job: a generation is VRAM-hungry and
        must not overlap an inference batch.

        Source: either raw bytes (browser upload the host forwarded) or a path
        the worker can already read. The artifact is ALWAYS zipped back, even
        when `data/avatars` is a shared symlink — one code path, and the host
        decides where it lands.
        """
        with self._gpu_work():
            work = None
            try:
                import json
                import shutil

                backend = request.backend or self.backend_name
                if backend not in _BACKENDS:
                    raise ValueError(
                        f"unknown backend {backend!r}; have {list(_BACKENDS)}")
                if not request.avatar_id:
                    raise ValueError("avatar_id is required")

                params = json.loads(request.params_json or "{}")

                work = tempfile.mkdtemp(prefix="lt_genavatar_")
                if request.source:
                    ext = os.path.splitext(request.filename or "")[1].lower() or ".mp4"
                    video_path = os.path.join(work, f"source{ext}")
                    with open(video_path, "wb") as fh:
                        fh.write(request.source)
                elif request.source_path:
                    video_path = request.source_path
                    if not os.path.isfile(video_path):
                        raise FileNotFoundError(video_path)
                else:
                    raise ValueError("source bytes or source_path is required")

                save_path = os.path.join(work, "avatars")

                if backend == "musetalk":
                    from avatars.musetalk.genavatar import generate_avatar
                    generate_avatar(
                        video_path=video_path,
                        avatar_id=request.avatar_id,
                        save_path=save_path,
                        bbox_shift=params.get("bbox_shift", 0),
                        extra_margin=params.get("extra_margin", 10),
                        parsing_mode=params.get("parsing_mode", "jaw"),
                        version=params.get("version", "v15"),
                    )
                elif backend == "wav2lip":
                    from avatars.wav2lip.genavatar import generate_avatar
                    generate_avatar(
                        video_path=video_path,
                        avatar_id=request.avatar_id,
                        save_path=save_path,
                        img_size=params.get("img_size", 256),
                        pads=params.get("pads", [0, 10, 0, 0]),
                        nosmooth=params.get("nosmooth", False),
                        face_det_batch_size=params.get("face_det_batch_size", 16),
                    )

                artifact_dir = os.path.join(save_path, request.avatar_id)
                if not os.path.isdir(artifact_dir):
                    raise RuntimeError(
                        f"genavatar produced no artifact dir at {artifact_dir}")

                files = []
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
                    for root, _dirs, names in os.walk(artifact_dir):
                        for name in names:
                            full = os.path.join(root, name)
                            rel = os.path.relpath(full, artifact_dir)
                            zf.write(full, rel)
                            files.append(rel)

                logger.info("[worker] GenerateAvatar %s/%s -> %d files (%.1f MB)",
                            backend, request.avatar_id, len(files),
                            len(buf.getvalue()) / 1e6)
                return inference_pb2.GenerateAvatarReply(
                    ok=True, msg=f"generated {len(files)} files",
                    archive=buf.getvalue(), frames=len(files))
            except Exception as exc:  # noqa: BLE001
                logger.exception("[worker] GenerateAvatar failed")
                return inference_pb2.GenerateAvatarReply(ok=False, msg=str(exc))
            finally:
                if work:
                    try:
                        import shutil as _sh
                        _sh.rmtree(work, ignore_errors=True)
                    except Exception:  # noqa: BLE001
                        pass


def serve(host: str, port: int, backend: Optional[str]) -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[("grpc.max_send_message_length", _MAX_MSG),
                 ("grpc.max_receive_message_length", _MAX_MSG),
                 # Keepalive tuning for a long-lived streaming session. The host
                 # pings every 20 s (see remote/client.py); with gRPC's default
                 # policy a server starts throttling that as "too many pings"
                 # and eventually sends GOAWAY ENHANCE_YOUR_CALM, which kills a
                 # live avatar session mid-stream:
                 #   Received a GOAWAY ... debug data "too_many_pings"
                 # Raise the tolerated ping rate and the minimum interval so the
                 # host's 20 s heartbeat is always legal.
                 ("grpc.keepalive_permit_without_calls", 1),
                 ("grpc.http2.max_pings_without_data", 0),
                 ("grpc.http2.min_ping_interval_without_data_ms", 5000),
                 ("grpc.http2.min_time_between_pings_ms", 5000)])
    inference_pb2_grpc.add_RemoteWorkerServicer_to_server(
        RemoteWorkerServicer(default_backend=backend), server)
    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info(f"[worker] listening on {addr} (backend={backend or 'lazy'})")
    server.wait_for_termination()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="LiveTalking remote GPU worker")
    # Loopback by default ON PURPOSE: the protocol is plaintext and
    # unauthenticated, so the box must not expose it. Reach it through an SSH
    # local-forward (`ssh -N -L 50051:127.0.0.1:50051 user@box`, which the host
    # does automatically in tunnel mode). Pass --host 0.0.0.0 only on a trusted
    # LAN where you have decided the exposure is acceptable.
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default 127.0.0.1: reachable only through "
                        "an SSH tunnel; use 0.0.0.0 on a trusted LAN)")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--backend", default=None,
                   choices=["musetalk", "wav2lip"],
                   help="pre-load this backend at startup (default: lazy)")
    p.add_argument("--check", action="store_true",
                   help="print the revision/device/backend this worker would "
                        "serve, then exit (used by the deploy preflight)")
    args = p.parse_args(argv)
    if args.check:
        # Proof that THIS interpreter can import everything the worker needs.
        # Importing torch alone is misleading: the worker's first two imports are
        # `grpc` and numpy, and a missing grpc is the most common first-deploy
        # failure. Printing each failure individually turns "RESTARTED=DEAD" into
        # a one-line answer.
        missing = []
        for mod in ("grpc", "numpy", "torch"):
            try:
                __import__(mod)
            except Exception as exc:  # noqa: BLE001
                missing.append(f"{mod}: {exc}")
        if missing:
            print("missing dependencies (this interpreter):")
            for line in missing:
                print("  " + line)
            return 2
        import torch
        say = (f"revision={_revision() or 'unknown'} "
               f"device={'cuda' if torch.cuda.is_available() else 'cpu'} "
               f"backend={args.backend or 'lazy'} "
               f"cwd={os.getcwd()} python={sys.executable}")
        if torch.cuda.is_available():
            say += f" gpu={torch.cuda.get_device_name(0)}"
        print(say)
        return 0
    serve(args.host, args.port, args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
