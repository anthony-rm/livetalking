"""Remote GPU worker — runs on the Windows/NVIDIA box.

Start it on the PC (same repo checkout, conda env with CUDA torch):

    python -m remote.worker --host 0.0.0.0 --port 50051 --backend musetalk

The Mac server talks to it via remote/client.py; nothing else here is exposed.
This process owns ALL GPU work: avatar inference (UNet+VAE / Wav2Lip) and the
base-video enhancer (RVM matting + Real-ESRGAN + GFPGAN). The Mac keeps TTS,
ASR feature extraction, paste-back, and streaming.

Only ONE backend + ONE avatar is resident at a time (the app is single-model by
design). Loading a different backend/avatar replaces the resident one.
"""
from __future__ import annotations

import argparse
import io
import os
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
        from avatars.musetalk_avatar import load_model
        self.model = load_model()
        logger.info("[worker] musetalk weights loaded")

    def avatar_artifacts(self, avatar_id: str):
        """Inference-only artifacts (latents). paste_back metadata stays on Mac."""
        import torch
        path = f"./data/avatars/{avatar_id}/latents.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return torch.load(path)

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
            audio_feature_batch = torch.from_numpy(whisper_batch).to(
                device=unet.device, dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
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
    def Health(self, request, context):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu = ""
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        return inference_pb2.HealthReply(
            ok=True, device=device, backend=self.backend_name,
            avatar_id=self.avatar_id, detail=gpu)

    def LoadModel(self, request, context):
        with self.lock:
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
        with self.lock:
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

    def InferBatch(self, request, context):
        with self.lock:
            try:
                if self.backend is None or self.backend.model is None:
                    raise RuntimeError("no model loaded; call LoadModel first")
                feats = [from_tensor(t) for t in request.audiofeat]
                pred = self.backend.infer(int(request.index),
                                          int(request.batch_size), feats)
                return inference_pb2.InferReply(frames=to_tensor(pred))
            except Exception as exc:  # noqa: BLE001
                logger.exception("[worker] InferBatch failed")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return inference_pb2.InferReply()

    def Unload(self, request, context):
        with self.lock:
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
        with self.lock:
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


def serve(host: str, port: int, backend: Optional[str]) -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[("grpc.max_send_message_length", _MAX_MSG),
                 ("grpc.max_receive_message_length", _MAX_MSG)])
    inference_pb2_grpc.add_RemoteWorkerServicer_to_server(
        RemoteWorkerServicer(default_backend=backend), server)
    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info(f"[worker] listening on {addr} (backend={backend or 'lazy'})")
    server.wait_for_termination()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="LiveTalking remote GPU worker")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--backend", default=None,
                   choices=["musetalk", "wav2lip"],
                   help="pre-load this backend at startup (default: lazy)")
    args = p.parse_args(argv)
    serve(args.host, args.port, args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
