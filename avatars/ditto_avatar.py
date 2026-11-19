###############################################################################
#  Ditto 数字人 — antgroup/ditto-talkinghead, wired as an in-process avatar
#
#  Data flow (all components are upstream's own, imported from the probe tree
#  via DITTO_HOME on sys.path):
#
#    PCM 20 ms chunks -> PcmASR (Ditto HuBERT, 5 feats / 0.2 s = 1 per frame)
#      -> ConditionHandler -> Audio2Motion (LMDM, online: overlap_v2=70, 10 steps)
#      -> MotionStitch -> WarpF3D + DecodeF3D (512²) -> numpy putback
#
#  Streaming law (mirrors upstream's audio2motion_worker exactly):
#  the LMDM window is 80 feature rows and SLIDES BY valid_clip_len (10) per
#  pass, keeping 70 rows of context; a pass emits the 10 new frames. The cond
#  row buffer is seeded with 80 zero rows (upstream seeds zeros in front of
#  the audio) so the first pass latches d0 immediately and real frames start
#  one batch later — TTFF ≈ 2 batches instead of a 3.2 s ramp.
#
#  Per-frame costs measured on the box (RTX 3060, cuDNN on, torch 2.9.1):
#  LMDM 31 ms amortized, stitch 1.6, warp 180.7, decode 84.0, putback ~39.
###############################################################################
from __future__ import annotations

import os

_EMO_TRACE = os.environ.get("LT_EMO_TRACE") == "1"
_XD_TRACE = os.environ.get("LT_XD_TRACE") == "1"
_XD_TARGET_ADJUST = float(os.environ.get("LT_XD_TARGET_ADJUST", "0"))
import pickle
import sys
import math
import time
import collections
import queue
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import cv2

from avatars.base_avatar import BaseAvatar
from avatars.audio_features.pcm import PcmASR, make_ditto_feat_fn
from avatars.ditto import artifact as art
from avatars.ditto import emotion as ditto_emotion
from registry import register
from utils.logger import logger
from utils.device import initialize_device

device = initialize_device()
logger.info('Using {} for ditto inference.'.format(device))

DITTO_HOME = os.environ.get("DITTO_HOME", os.path.expanduser("~/ditto-probe"))
DITTO_WEIGHTS = os.environ.get(
    "DITTO_WEIGHTS", os.path.expanduser("~/ditto-weights/ditto_pytorch"))
DITTO_CFG = os.environ.get(
    "DITTO_CFG",
    os.path.expanduser("~/ditto-weights/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"))

#: Online cfg (runtime kwargs — there is no PyTorch online cfg FILE). Measured
#: 313 ms/pass for 10 frames = 31 ms/frame, inside the 400 ms/pass budget.
#: T9: overlap_v2 / sampling_timesteps are env-overridable for A/B —
#: DITTO_OVERLAP_V2 (default 70) widens valid_clip_len (15/20) so the fixed
#: ~90 ms LMDM pass amortizes over more frames; DITTO_LMDM_STEPS (default 10)
ONLINE_KWARGS = dict(
    overlap_v2=int(os.environ.get("DITTO_OVERLAP_V2", "70")),
    sampling_timesteps=int(os.environ.get("DITTO_LMDM_STEPS", "10")),
    online_mode=True)

#: HuBERT feature dim (PcmASR.FEAT_DIM). Ditto's own checkpoint; nothing else
#: may substitute (§4 of the skill reference).
FEAT_DIM = 1024


#: Sentinel returned by inference_batch for frames whose pixels are still in
#: an async GPU->host copy. paste_back_frame resolves it via _take_staged_frame.
_STAGED_FRAME = object()

#: LT_STAGE_TRACE=1 logs per-stage batch averages (lmdm/stitch/warp/dec/stage)
#: every 25 batches — the live-vs-offline cost attribution instrument.
_STAGE_TRACE = os.environ.get("LT_STAGE_TRACE") == "1"

#: Give the TRT LMDM engine its own CUDA stream (see _enable_lmdm_async_stream).
#: 0 = off (blocking infer, current behaviour), 1 = on (async, concurrent with
#: the renderer). Measured gain target: 14.0 -> 16.5-17.5 fps at 512.
_LMDM_ASYNC = os.environ.get("LT_LMDM_ASYNC", "1") not in ("0", "false", "no")

#: Idle lane for SINGLE-IMAGE artifacts (see DittoReal.idle_frame). A still
#: image replays as a frozen photograph, so the silent branch renders a
#: synthesized breathing loop instead. `LT_IDLE_MOTION=0` restores the still.
_IDLE_MOTION = os.environ.get("LT_IDLE_MOTION", "1") not in ("0", "false", "no")
#: Displayed frames per RENDERED idle frame. Idle needs no 25 fps: a breath
#: read at ~6 fps looks the same and costs a quarter of the GPU.
_IDLE_REUSE = max(1, int(os.environ.get("LT_IDLE_REUSE", "4")))
#: Rendered frames in one idle loop (period of the breathing sinusoid).
_IDLE_LOOP = max(2, int(os.environ.get("LT_IDLE_LOOP", "48")))


def _enable_lmdm_async_stream(a2m) -> bool:
    """Give the TRT LMDM engine its own CUDA stream and use the async path.

    WHY THIS IS THE REAL LEVER (measured on the box, 2026-11-18):
    TRTWrapper.infer() ends with a BLOCKING cudaDeviceToHost copy of pred_noise
    and x_start — and that blocking copy synchronises the WHOLE default
    stream. The GPU worker runs warp+decode on the same default stream, so
    every LMDM step stalls for the entire renderer's queued work. That is why
    the live lane reports lmdm=172 ms/batch while the same pass measured
    ISOLATED is 90 ms (2x inflation), and why the live batch (695 ms) exceeds
    the sum of the isolated parts (90 + 550 = 640 ms) by ~55 ms.

    TRTWrapper.infer_async(stream) already exists upstream — same kernels,
    async H2D + execute_async_v3 + async D2H on the given stream. Switching
    the LMDM to it on a PRIVATE stream makes its 90 ms of kernels overlap with
    the renderer's 550 ms instead of serialising behind them.

    Output is bit-identical: the sync below waits only for the LMDM's own
    stream, and the inference thread reads the same buffer it would have
    read from the blocking path. No numerics change, no GIL change.
    """
    if not _LMDM_ASYNC:
        return False
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    # Object chain: Audio2Motion.lmdm -> LMDM.model -> TRTWrapper.
    lmdm = getattr(a2m, "lmdm", None)
    wrapper = getattr(lmdm, "model", None)
    if lmdm is None or wrapper is None \
            or not hasattr(wrapper, "infer_async"):
        return False
    a2m._lmdm_stream = torch.cuda.Stream(device=device)
    a2m._lmdm_async = True
    logger.info("[ditto] LMDM private stream on (per-step waits scoped to "
                "it; the renderer keeps the default stream)")
    return True


def _patch_lmdm_call_np(a2m) -> None:
    """Swap the LMDM's per-step engine call to the async path on its private
    stream, and scope each wait to that stream alone.

    Upstream's _one_step (tensorrt branch):
        model.setup({...})                  # cudaFree+cudaMalloc per call
        model.infer()                       # BLOCKING copies on the NULL stream
    Two hidden serialization sources: (1) blocking cudaMemcpy is issued on
    the legacy NULL stream, which every torch-created stream synchronises
    with — so each step's D2H waits for the renderer's whole queued backlog;
    (2) setup()'s cudaFree/cudaMalloc pair is an implicit device-wide sync.

    Replacement, per DDIM step:
      prime once  : full setup() with the first real arrays — allocates the
                    device buffers, binds addresses, latches shapes (all
                    constant for the run).
      then        : refresh the four HOST input arrays in place (same
                    pointers are re-read at copy time; no free/malloc, no
                    rebind), infer_async(private stream), synchronize THAT
                    stream only, read pred_noise/x_start.
    The DDIM recurrence itself (upstream's _call_np) is untouched: the
    sync per step is unavoidable — the host update consumes each step's
    output — but it no longer drags the renderer's stream with it.
    """
    stream = a2m._lmdm_stream
    lmdm = a2m.lmdm
    wrapper = lmdm.model
    primed = [False]

    def _one_step_async(x, cond_frame, cond, time_cond):
        if not primed[0]:
            wrapper.setup({"x": x, "cond_frame": cond_frame,
                           "cond": cond, "time_cond": time_cond})
            primed[0] = True
        else:
            for name, data in (("x", x), ("cond_frame", cond_frame),
                               ("cond", cond), ("time_cond", time_cond)):
                wrapper.buffer[name][0] = np.ascontiguousarray(data)
        wrapper.infer_async(stream.cuda_stream)
        stream.synchronize()
        return wrapper.buffer["pred_noise"][0], wrapper.buffer["x_start"][0]

    lmdm._one_step = _one_step_async
    logger.info("[ditto] LMDM engine call: infer_async, sync scoped to the "
                "private stream")


class _GpuJob:
    """A closure to run ON the GPU worker thread (not a batch of frames).

    Needed because the engines and the batched decoder hold static IO buffers:
    a second thread calling warp/decode concurrently would corrupt them. Idle
    frame rendering therefore rides the one existing GPU thread instead of
    taking a lock the speech lane would have to share.
    """

    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn


class _LocalGpuPipeline:
    """Two-thread pipelining for ditto's inference_batch, local-process.

    Live 512 attribution (LT_STAGE_TRACE, batch of 10): lmdm 172 + stitch
    112 ms on the CPU side vs warp 253 + decode 270 ms on the GPU side.
    base_avatar runs them serially, so a batch costs ~810 ms. The GIL probe
    at scratch:gil_overlap_probe.py showed a CPU-only Python thread costs
    the GPU thread ~10%, i.e. TRT/torch release the GIL — so the two halves
    can overlap: the GPU worker runs warp/decode/stage for batch N while
    the inference thread lmdm+stitches batch N+1. Steady state falls to
    the GPU half (~525 ms/batch, ~19 fps) instead of the serial sum.

    Exactly ONE worker thread: batches serialize on the GPU anyway (one
    engine pair), and a second submitter would only grow the token deque.
    Throughput contract: submit() never blocks on compute (bounded queue of
    1), result() blocks until that token's frames are staged.
    """

    def __init__(self, owner, batch_size, cpu_fn, gpu_fn):
        self._owner = owner
        self._cpu_fn = cpu_fn
        self._gpu_fn = gpu_fn
        self._q = queue.Queue(maxsize=1)
        self._quit = threading.Event()
        self._thread = threading.Thread(target=self._worker,
                                        name="ditto-gpu", daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._quit.is_set():
            try:
                token, payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if isinstance(payload, _GpuJob):
                    payload.fn()
                else:
                    self._gpu_fn(payload)
            except BaseException as e:  # surface at result(); keep serving
                token[1] = e
            finally:
                token[2].set()

    def submit(self, index, batch_size, audiofeat_batch, **kwargs):
        """CPU half on the CALLER's thread (GIL-bound), GPU half queued.

        Returns a token; result(token) blocks until the GPU half has staged
        the batch's pixels. The one-slot queue is the backpressure: with
        keep_in_flight=1 base_avatar holds exactly one submitted batch, so
        at most one GPU half waits — the GPU runs batch N while the CPU
        runs lmdm+stitch of batch N+1.
        """
        token = [False, None, threading.Event()]
        payload = self._cpu_fn(index, batch_size, audiofeat_batch)
        self._q.put((token, payload))
        return token

    def run_on_gpu(self, fn, timeout=30.0):
        """Run `fn` on the GPU worker thread and wait for it.

        Used by the idle lane (ditto's `idle_frame`), which renders a crop
        outside the batch contract. Serialising through the ONE worker keeps
        the engines' static buffers single-owner; the wait is why the idle
        path caches its result instead of rendering per displayed frame.
        Errors re-raise here, on the caller's thread.
        """
        token = [False, None, threading.Event()]
        self._q.put((token, _GpuJob(fn)))
        if not token[2].wait(timeout):
            raise TimeoutError("ditto gpu job timeout")
        if token[1] is not None:
            raise token[1]
        return token

    def result(self, token, timeout=None):
        """Block until submit()'s work is done; re-raise worker errors here.

        On error the worker drained its OWN partial staging (see _host) so
        the _staged ring stays aligned with emission order — a failed batch
        resolves to None frames and _take_staged_frame is never called for
        it, keeping later batches' slots in place.
        """
        if not token[2].wait(timeout or 3600):
            raise TimeoutError("ditto gpu pipeline result timeout")
        if token[1] is not None:
            raise token[1]
        return [_STAGED_FRAME] * self._owner.batch_size

    def stop(self):
        self._quit.set()
        self._thread.join(timeout=2)


def _ensure_ditto_home() -> None:
    if DITTO_HOME not in sys.path:
        sys.path.insert(0, DITTO_HOME)


def _install_cuda_shim() -> None:
    """cuda-python 13 moved `from cuda import cuda, cudart, nvrtc` into
    `cuda.bindings.{driver,runtime}` — TRTWrapper still imports the legacy
    names. Idempotent; no-op if the legacy layout is present."""
    try:
        import cuda  # noqa: F401
        from cuda import cuda as _c, cudart as _r, nvrtc as _n  # noqa: F401
        return  # legacy layout, nothing to do
    except ImportError:
        pass
    try:
        import types
        import cuda.bindings.driver as _drv
        import cuda.bindings.runtime as _rt
        m = types.ModuleType("cuda.cuda"); m.__dict__.update(_drv.__dict__)
        sys.modules["cuda.cuda"] = m
        m2 = types.ModuleType("cuda.cudart"); m2.__dict__.update(_rt.__dict__)
        sys.modules["cuda.cudart"] = m2
        m3 = types.ModuleType("cuda.nvrtc"); m3.__dict__.update(_rt.__dict__)
        sys.modules["cuda.nvrtc"] = m3
    except ImportError:
        pass


def load_model(_arg=None):
    """Load every Ditto torch component once (musetalk-style: no arguments).

    The six aux/ORT models stay on CPU by cfg — they are tiny and their CUDA
    EP faulted this box's dxg driver once; torch models go to the resolved
    device. `parse_cfg` MUTATES the dicts it reads, so the device patch is
    applied to the raw pkl first and the file is parsed once.
    """
    _ensure_ditto_home()
    import pickle as _pkl
    from core.atomic_components.cfg import parse_cfg
    from core.atomic_components.condition_handler import ConditionHandler
    from core.atomic_components.audio2motion import Audio2Motion
    from core.atomic_components.motion_stitch import MotionStitch
    from core.atomic_components.warp_f3d import WarpF3D
    from core.atomic_components.decode_f3d import DecodeF3D

    dev = "cuda" if device.type == "cuda" else str(device)
    ORT_CPU = {"insightface_det_cfg", "landmark106_cfg", "landmark203_cfg",
               "landmark478_cfg", "hubert_cfg", "wavlm_cfg"}
    with open(DITTO_CFG, "rb") as f:
        raw = _pkl.load(f)
    for k, v in raw.get("base_cfg", {}).items():
        if isinstance(v, dict) and "device" in v:
            v["device"] = "cpu" if k in ORT_CPU else dev
    if isinstance(raw.get("audio2motion_cfg"), dict) and \
            "device" in raw["audio2motion_cfg"]:
        raw["audio2motion_cfg"]["device"] = dev
    with open(DITTO_CFG, "wb") as f:
        _pkl.dump(raw, f)

    (reg_cfg, cond_cfg, lmdm_cfg, stitch_cfg, warp_cfg, dec_cfg,
     w2f_cfg, default_kwargs) = parse_cfg(DITTO_CFG, DITTO_WEIGHTS)

    # TRT decoder lane: if a TensorRT engine for the decoder exists (built for
    # THIS box — TRT engines are GPU+version specific), swap it in. Upstream's
    # Decoder class already dispatches on the .engine extension. The warp
    # network cannot go TRT without the grid_sample_3d plugin, which upstream
    # ships compiled against TRT 8; on a TRT-10 stack only the decoder is
    # portable. Measured gain here: 75 ms -> 31 ms per frame.
    engine_cfg = os.environ.get("DITTO_DECODER_ENGINE", "")
    if not engine_cfg:
        cand = os.path.join(os.path.dirname(DITTO_WEIGHTS.rstrip("/")),
                            "ditto_trt_Ampere_Plus", "decoder_fp16_1013.engine")
        if os.path.exists(cand):
            engine_cfg = cand
    if engine_cfg and os.path.exists(engine_cfg):
        _install_cuda_shim()
        logger.info("[ditto] decoder engine: %s", engine_cfg)
        dec_cfg = dict(dec_cfg, model_path=engine_cfg)

    # Hybrid TRT warp lane (plugin-free, TRT 10): three sub-engines split at
    # the two 5-D GridSample nodes, with the grid samples run in torch. Needed
    # because the prebuilt warp engine embeds the TRT-8 grid_sample_3d plugin.
    # Measured: 124.6 ms -> 24.4 ms per frame (5.1x), same numerics.
    #
    # render_size picks the engine set by its FEATURE EDGE (E = render/8 —
    # the SPADE decoder upsamples 8x; TRTWarpHybrid reports E from the
    # engine's segC input shape):
    #   render 512 -> E=64 (warp+decode ~50 ms/frame)
    #   render 256 -> E=32 (warp+decode ~25 ms/frame; paste matrix scaled)
    # Resolution cost is real but bounded: the 512 path is the same face at
    # 4x the pixels; the 256 render covers the same crop region (proven by
    # PSNR 23 dB against the 512 render downsampled, and by the paste bbox
    # equality at Mc = [M_lin * (512/pred_w) | b]).
    warp = None
    render_size = os.environ.get("DITTO_RENDER_SIZE", "auto").strip().lower()
    cfg_base = os.path.dirname(DITTO_WEIGHTS.rstrip("/"))
    if render_size == "auto":
        # Prefer the 256 lane when its engines exist: same face, quarter the
        # render pixels, and it is what keeps delivery at the 25 fps audio
        # rate on this box. An explicit 512 forces the full-resolution lane.
        cand = os.path.join(cfg_base, "warp_r256_engines")
        render_size = 256 if os.path.exists(
            os.path.join(cand, "hyb_segB.engine")) else 512
    else:
        render_size = int(render_size)
    if render_size not in (256, 512):
        raise ValueError(f"DITTO_RENDER_SIZE must be auto, 256 or 512, "
                         f"got {render_size}")
    hybrid_dir = os.environ.get("DITTO_WARP_ENGINE_DIR", "")
    dec_b10_override = os.environ.get("DITTO_DECODE_B10_ENGINE", "")
    # INT8 lane: opt-in via DITTO_INT8=1. Engines live in warp_int8_engines/
    # (entropy-calibrated, FP32 I/O — drop-in for the wrappers). Auto-detect
    # both engines; fail loudly if only one exists (mixed lanes are
    # untested numerics).
    if os.environ.get("DITTO_INT8", "").strip().lower() in ("1", "true", "yes"):
        int8_dir = os.path.join(cfg_base, "warp_int8_engines")
        warp_ok = os.path.exists(os.path.join(int8_dir, "hyb_segB.engine"))
        dec_engine = os.path.join(int8_dir, "decoder_b10_int8.engine")
        if not (warp_ok and os.path.exists(dec_engine)):
            raise RuntimeError(
                f"DITTO_INT8=1 but the int8 engine set is incomplete in "
                f"{int8_dir} (warp segs: {warp_ok}, decoder: "
                f"{os.path.exists(dec_engine)}) — run build_int8.py")
        hybrid_dir = int8_dir
        dec_b10_override = dec_engine
        logger.info("[ditto] INT8 lane: %s", int8_dir)
    if not hybrid_dir and render_size == 256:
        cand = os.path.join(cfg_base, "warp_r256_engines")
        if os.path.exists(os.path.join(cand, "hyb_segB.engine")):
            hybrid_dir = cand
    if not hybrid_dir:
        cand = os.path.join(cfg_base, "warp_hybrid_engines")
        if os.path.exists(os.path.join(cand, "hyb_segB.engine")):
            hybrid_dir = cand
    if hybrid_dir:
        _install_cuda_shim()
        from avatars.ditto.trt_warp_hybrid import TRTWarpHybrid
        warp = TRTWarpHybrid(hybrid_dir, device=dev)
        engine_edge = warp.feature_edge
        if render_size == 256 and engine_edge != 32:
            raise RuntimeError(
                f"DITTO_RENDER_SIZE=256 but warp engines have feature edge "
                f"{engine_edge} (need 32) — build warp_r256_engines (scratch "
                f"export_r256.py) or unset the override")
        if render_size == 512 and engine_edge not in (64, None):
            raise RuntimeError(
                f"DITTO_RENDER_SIZE=512 but warp engines have feature edge "
                f"{engine_edge} (need 64)")
        logger.info("[ditto] warp: TRT-10 hybrid engines from %s", hybrid_dir)

    # TRT LMDM lane: the online DDIM sampler stays in numpy on the host; each
    # diffusion step becomes one engine call (upstream's LMDM already speaks
    # TRTWrapper: setup({x, cond_frame, cond, time_cond}) -> pred_noise/x_start).
    # Measured: 33.9 -> 8.75 ms per step (3.9x), rel diff vs pytorch ~1e-5.
    lmdm_engine = os.environ.get("DITTO_LMDM_ENGINE", "")
    if not lmdm_engine:
        cand = os.path.join(os.path.dirname(DITTO_WEIGHTS.rstrip("/")),
                            "lmdm_v0.4_hubert_fp16_trt10.engine")
        if os.path.exists(cand):
            lmdm_engine = cand
    if lmdm_engine and os.path.exists(lmdm_engine):
        _install_cuda_shim()
        logger.info("[ditto] lmdm engine: %s", lmdm_engine)
        lmdm_cfg = dict(lmdm_cfg, model_path=lmdm_engine)

    # Batched DECODE lane (static B=10): the LMDM emits the whole 10-frame
    # window per pass, so the decoder runs once per window instead of ten
    # times (measured 31.8 -> 25.9 ms/frame). The warp stays on the per-frame
    # hybrid: a B=10 warp engine was built and measured SLOWER (32.3 vs 24.5
    # ms/frame) — the dense-motion hourglass is already compute-saturated at
    # B=1, batching only removes launch overhead. Mixed = 50.4 ms/frame.
    dec_b10 = os.environ.get("DITTO_DECODE_B10_ENGINE", "")
    if dec_b10_override:
        dec_b10 = dec_b10_override
    if not dec_b10 and render_size == 256:
        cand = os.path.join(cfg_base, "warp_r256_engines",
                            "decoder_b10_r256_fp16.engine")
        if os.path.exists(cand):
            dec_b10 = cand
    if not dec_b10:
        cand = os.path.join(cfg_base, "warp_b10_engines",
                            "decoder_b10_fp16.engine")
        if os.path.exists(cand):
            dec_b10 = cand
    if dec_b10 and os.path.exists(dec_b10):
        _install_cuda_shim()
        from avatars.ditto.trt_decode_batch import TRTDecodeBatch
        dec = TRTDecodeBatch(dec_b10, device=dev)
        if render_size == 256 and tuple(dec.in_buf.shape)[2:] != (32, 32):
            raise RuntimeError(
                f"DITTO_RENDER_SIZE=256 but decode engine expects "
                f"{tuple(dec.in_buf.shape)} (need (B,256,32,32))")
        logger.info("[ditto] decode: batched engine %s", dec_b10)
        return {
            "cond": ConditionHandler(**cond_cfg),
            "a2m": Audio2Motion(lmdm_cfg),
            "stitch": MotionStitch(stitch_cfg),
            "warp": warp or WarpF3D(warp_cfg),
            "dec": dec,
            "default_kwargs": default_kwargs,
        }

    return {
        "cond": ConditionHandler(**cond_cfg),
        "a2m": Audio2Motion(lmdm_cfg),
        "stitch": MotionStitch(stitch_cfg),
        "warp": warp or WarpF3D(warp_cfg),
        "dec": DecodeF3D(dec_cfg),
        "default_kwargs": default_kwargs,
    }


#: x_s_info per-key shapes as MotionExtractor produces them (the artifact
#: stores them flat; the reader restores these exact shapes).
_X_S_SHAPES = {"kp": (1, 63), "exp": (1, 63), "pitch": (1, 66),
               "yaw": (1, 66), "roll": (1, 66), "t": (1, 3), "scale": (1,)}


def _reshape_x_s(flat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in flat.items():
        shape = _X_S_SHAPES.get(k)
        if shape is None or int(np.prod(shape)) != v.size:
            shape = (1, v.size)
        out[k] = v.reshape(shape).astype(np.float32)
    return out


def load_avatar(avatar_id: str):
    """Read + validate the artifact. Raises ArtifactError loudly on mismatch."""
    avatar_dir = os.path.join(".", "data", "avatars", avatar_id)
    marker = art.validate(avatar_dir)
    n = int(marker["frame_count"])

    frame_list, x_s_list, f_s_list = [], [], []
    for i in range(n):
        img = cv2.imread(art.full_img_path(avatar_dir, i))
        if img is None:
            raise art.ArtifactError(
                f"{avatar_dir}: full_imgs/{i:08d}.png unreadable")
        frame_list.append(img)
        x_s_list.append(_reshape_x_s(art.read_x_s_info(avatar_dir, i)))
        f_s_list.append(art.read_f_s(avatar_dir, i))     # fp16 (32,16,64,64)
    with open(os.path.join(avatar_dir, "M_c2o.pkl"), "rb") as f:
        M_c2o_list = [np.asarray(m, dtype=np.float32) for m in pickle.load(f)]
    if len(M_c2o_list) != n:
        raise art.ArtifactError(
            f"M_c2o.pkl has {len(M_c2o_list)} entries, artifact says {n}")
    # (None, None) on pre-eye-state artifacts; __init__ falls back to zeros.
    eye_open, eye_ball = art.read_eye_state(avatar_dir)
    return marker, frame_list, x_s_list, f_s_list, M_c2o_list, eye_open, eye_ball


@torch.no_grad()
def warm_up(batch_size, model):
    """Prime the GPU components once so the first live batch pays no warmup."""
    _ensure_ditto_home()
    from core.atomic_components.motion_stitch import transform_keypoint

    parts = model
    warp = parts["warp"]
    # Seed the zero volume with the ENGINE's feature edge so the warmup
    # exercises the lane actually wired (512 -> E=64, 256 -> E=32).
    edge = getattr(warp, "feature_edge", None)
    zero = np.zeros((1, 32, 16, edge or 64, edge or 64), dtype=np.float32)
    x_s_info = {k: np.zeros(s, dtype=np.float32) for k, s in _X_S_SHAPES.items()}
    x_s = transform_keypoint(x_s_info)          # dict -> (bs, num_kp, 3)
    f_3d = parts["warp"](zero, x_s, x_s.copy())
    parts["dec"](f_3d)
    logger.info("[ditto] warmup done (warp+decode primed)")


def _shrink_f_s_list(f_s_list, edge: int):
    """Bilinear-downsample the stored 64-edge volumes to `edge` (256 render ->
    32). Registration always writes 64-edge f_s (the extractor resizes its
    input to 256^2 regardless of crop dsize), so the shrink is a pure
    resolution choice, done once per avatar load.

    Samples per frame: N*C*D downsampled H*W through one interpolate; done on
    the torch default device so a cuda box pays ~3 ms, and (~75 frames of
    4 MB fp32) fits any GPU. Returns the same length list, fp16 as stored."""
    import torch.nn.functional as TF  # resolved lazily; pyright misses the subimport
    src = np.stack(f_s_list)                        # (N,32,16,64,64) fp16
    t = torch.from_numpy(src.astype(np.float32))
    n, c, d, h, w = t.shape
    t = t.permute(0, 1, 2, 3, 4).reshape(n * c * d, 1, h, w)
    t = TF.interpolate(t, size=(edge, edge), mode="bilinear",
                       align_corners=False)
    t = t.reshape(n, c, d, edge, edge)
    out = t.numpy().astype(np.float16)
    return [out[i] for i in range(n)]


@register("avatar", "ditto")
class DittoReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        self.parts = model
        (self.marker, self.frame_list_cycle, self.x_s_list,
         self.f_s_list, self.M_c2o_list,
         eye_open, eye_ball) = avatar
        self.n_frames = len(self.frame_list_cycle)

        self.asr = PcmASR(opt, self, self._make_feat_fn())
        self.asr.warm_up()

        dkw = self.parts["default_kwargs"]
        self.ch = self.parts["cond"]
        if eye_open is not None:
            # Real per-frame eyelid/gaze measured at registration; the row
            # order matches x_s_info/frame order (ConditionHandler indexes it
            # with _mirror_index against the same global idx).
            eye_open_lst = [eye_open[i][None] for i in range(self.n_frames)]
            eye_ball_lst = [eye_ball[i][None] for i in range(self.n_frames)]
        else:
            eye_open_lst = [np.zeros((1, 2), np.float32)] * self.n_frames
            eye_ball_lst = [np.zeros((1, 6), np.float32)] * self.n_frames
        ch_info = {
            "x_s_info_lst": self.x_s_list,
            "sc": self.x_s_list[0]["kp"].reshape(-1),
            "eye_open_lst": eye_open_lst,
            "eye_ball_lst": eye_ball_lst,
        }
        self.ch.setup(ch_info, dkw["emo"], eye_f0_mode=dkw["eye_f0_mode"])

        self.a2m = self.parts["a2m"]
        self.a2m.setup(self.ch.x_s_info_0,
                       fix_kp_cond=dkw["fix_kp_cond"],
                       fix_kp_cond_dim=dkw["fix_kp_cond_dim"],
                       v_min_max_for_clip=dkw["v_min_max_for_clip"],
                       smo_k_d=dkw["smo_k_d"],
                       **ONLINE_KWARGS)

        # ── LMDM async stream (see _enable_lmdm_async_stream) ─────────────
        # Must run AFTER setup() (setup binds the engine context; the stream
        # is per-engine, not per-context, so it can be created here).
        _enable_lmdm_async_stream(self.a2m)
        if getattr(self.a2m, "_lmdm_async", False):
            _patch_lmdm_call_np(self.a2m)

        self.stitch = self.parts["stitch"]
        self.stitch.setup(
            N_d=-1, use_d_keys=dkw["use_d_keys"], relative_d=dkw["relative_d"],
            drive_eye=dkw["drive_eye"], delta_eye_arr=dkw["delta_eye_arr"],
            delta_eye_open_n=dkw["delta_eye_open_n"],
            fade_out_keys=dkw["fade_out_keys"], fade_type=dkw["fade_type"],
            flag_stitching=dkw["flag_stitching"],
            is_image_flag=(self.n_frames == 1),
            x_s_info=self.x_s_list[0], d0=None, ch_info=None,
            overall_ctrl_info=dkw["overall_ctrl_info"])

        self.warp = self.parts["warp"]
        self.dec = self.parts["dec"]

        # render-size lane: engines carry the feature edge (64 -> render
        # 512, 32 -> render 256; render = edge * 8 — the SPADE decoder
        # upsamples 8x). Artifacts ALWAYS store 64-edge f_s — the extractor
        # resized its input to 256^2 at registration regardless of crop
        # dsize — so the 256 lane just shrinks the volumes at load.
        edge = getattr(self.warp, "feature_edge", None) \
            or int(self.marker["render_size"]) // 8
        marker_edge = int(self.marker["render_size"]) // 8
        self.render_edge = edge
        if edge > marker_edge:
            raise art.ArtifactError(
                f"artifact render_size={self.marker['render_size']} cannot "
                f"drive {edge * 8}-render engines; re-register the avatar "
                f"at a larger crop")
        if edge < marker_edge:
            logger.info("[ditto] render %d lane: shrinking f_s %d -> %d edge "
                        "(%d frames)", edge * 8, marker_edge, edge,
                        self.n_frames)
            self.f_s_list = _shrink_f_s_list(self.f_s_list, edge)

        # ── streaming state (§3: why a stateless batch API cannot host Ditto) ──
        # Cond row buffer, SEEDED with one window of silence so the first pass
        # latches d0 without waiting for 3.2 s of speech (upstream seeds zeros
        # in front of the audio for the same reason).
        self._audio_feat = np.zeros((self.a2m.seq_frames, FEAT_DIM), np.float32)
        # Same seed length as _audio_feat: the zero rows are silence, so they
        # carry no tag. Starting _emo_rows empty shifted every tag 80 rows
        # (3.2 s) ahead of the window that read it — _window_emotion was
        # resolving windows against the NEXT sentence's emotion.
        self._emo_rows: List = [None] * self.a2m.seq_frames
        self._local_idx = 0        # window start inside _audio_feat
        self._global_idx = 0       # absolute feature-frame counter (cond idx)
        self._res_kp_seq = None    # fused LMDM output
        self._valid_start = None   # first un-emitted frame in _res_kp_seq
        self._x_d_pending: List[dict] = []
        self._gen_frame_idx = 0
        # Bounded pending queue: in drop mode the LMDM emits motion at the
        # FEATURE arrival rate (25 rows/s, one per 40 ms of audio) while the
        # renderer consumes at the render rate (~18 fps). A FIFO that is never
        # trimmed grows ~7 frames/s during sustained speech, and the picture
        # ends up showing motion for audio the track played seconds earlier —
        # the "lips don't track, huge latency" report. Trim oldest here: their
        # audio already left the track, so rendering them is pure lag. The
        # newest entries correspond to the audio being heard NOW.
        self._xd_queue_max = int(os.environ.get("LT_XD_QUEUE_MAX", "20"))
        # Absolute feature-row counter for the NEXT emitted x_d entry. Each
        # feature row covers 2 audio chunks (40 ms), so entry i emitted from a
        # pass starting at _emit_row covers chunk ~2*(_emit_row + i) minus the
        # ASR right-context lead. Tagging entries lets _cpu_half pick the one
        # whose audio is being played NOW instead of FIFO — that is the only
        # lip-sync fix that works when the render rate is below the audio
        # rate, and it is what the earlier frame-level drop got wrong (it
        # compared rendered frames against a pointer advancing faster than
        # the renderer and dropped everything -> freeze).
        self._emit_row = 0
        self._lmdm_ms = 0.0
        self._lmdm_passes = 0
        self._stage_n = 0
        self._t_lmdm_ms = self._t_stitch_ms = 0.0
        self._t_warp_ms = self._t_dec_ms = self._t_stage_ms = 0.0
        self._idle_xd: Optional[dict] = None
        self._mask_cache: Optional[np.ndarray] = None
        self._emo_override = None   # session-sticky emotion (set_emotion)
        self._idle_tick = 0       # idle-lane frame counter (see idle_frame)
        self._idle_cache: Dict[int, np.ndarray] = {}   # rendered idle crops
        self._idle_failed: Optional[str] = None   # throttles the idle warning
        self._last_emo = None     # for the [emo] trace log
        self._emo_trace_n = 0     # throttle: log the first N tagged windows
        self._xd_trace_n = 0      # throttle for the [xd] pick trace
        # bbox + crop-matrix per (frame idx, render size), so paste-back never
        # re-derives the warped-mask bounds; 75 frames -> ~75 tiny tuples.
        self._paste_geom_cache: dict = {}
        # ── async GPU->host handoff (pinned ring + CUDA events) ──────────────
        # .cpu() on the decode output stalled the inference thread ~27 ms/frame
        # while the GPU idled. Instead the uint8 conversion runs ON GPU and the
        # copy lands in pinned memory asynchronously; consumers wait only when
        # they actually read the pixels (paste_back_frame). Emission order is
        # preserved by construction: the deque is FIFO and the markers carry
        # the frame's slot position implicitly.
        self._staged: "collections.deque[dict]" = collections.deque()

        # ── two-thread lane (see _LocalGpuPipeline) ─────────────────────────
        # Only when a batched decode engine exists AND CUDA is present: the
        # CPU fallback path shares self.warp/dec with the tests, which must
        # stay synchronous.
        self.infer_pipeline = None
        if device.type == "cuda" and hasattr(self.dec, "batch"):
            if hasattr(self.warp, "prime_f_s"):
                # T14: stage the cyclic f_s pool on device once (~630 MB at
                # edge 64) — kills the ~84 MB/batch pageable H2D + the
                # per-frame astype(fp32) in _cpu_half.
                self.warp.prime_f_s(
                    [f.astype(np.float32) for f in self.f_s_list])
            self.infer_pipeline = _LocalGpuPipeline(
                self, self.batch_size, self._cpu_half, self._gpu_half)

    def _warp_pool_staged(self):
        return getattr(self.warp, "_f_s_pool", None) is not None \
            and hasattr(self.warp, "run_batch_dev")

    def _cpu_half(self, index, batch_size, audiofeat_batch):
        """lmdm + stitch + prep — runs on the inference thread (GIL-bound).

        Returns the (x_s, x_d, f_s) batches for the GPU half; the GPU worker
        appends its pixels to _staged before resolving this token, so
        emission order equals _take_staged_frame's consumption order.
        """
        emos = None
        if isinstance(audiofeat_batch, tuple) and len(audiofeat_batch) == 2:
            audiofeat_batch, emos = audiofeat_batch
        feats = np.asarray(audiofeat_batch, dtype=np.float32)
        trace = _STAGE_TRACE
        t_pre = time.perf_counter() if trace else 0.0
        self._drive_lmdm(feats, emos)
        if trace:
            self._t_lmdm_ms += (time.perf_counter() - t_pre) * 1000
            t_pre = time.perf_counter()
        x_s_batch, x_d_batch, f_s_batch = [], [], []
        for _ in range(batch_size):
            x_d = self._pick_xd()
            idx = _mirror(self._gen_frame_idx, self.n_frames)
            x_s, x_d = self.stitch(self.x_s_list[idx], x_d)
            x_s_batch.append(x_s)
            x_d_batch.append(x_d)
            f_s_batch.append(idx if self._warp_pool_staged()
                              else self.f_s_list[idx].astype(np.float32))
            self._gen_frame_idx += 1
        if trace:
            self._t_stitch_ms += (time.perf_counter() - t_pre) * 1000
            if self._stage_n % 25 == 0 and self._stage_n > 0:
                logger.info(
                    "[ditto] cpu: lmdm=%.1f stitch=%.1f (avg ms)",
                    self._t_lmdm_ms / self._stage_n,
                    self._t_stitch_ms / self._stage_n)
        return x_s_batch, x_d_batch, f_s_batch

    def _gpu_half(self, payload):
        """warp+decode+stage for one batch — runs on the pipeline worker.

        Reads only immutable-per-run state (f_s_list, warp, dec, render_edge),
        appends to the FIFO _staged ring; order-safe against paste_back_frame.
        """
        with torch.no_grad():
            x_s_batch, x_d_batch, f_s_batch = payload
            e = self.render_edge
            trace = _STAGE_TRACE
            t_pre = time.perf_counter() if trace else 0.0
            if self._warp_pool_staged():
                # T14: f_s comes from the device pool by frame index — no
                # 8.4 MB H2D per chain, no numpy astype on this thread.
                f_3d = self.warp.run_batch_dev(
                    [(f_s, x_s, x_d) for x_s, x_d, f_s
                     in zip(x_s_batch, x_d_batch, f_s_batch)])
                f_3d = torch.stack(f_3d).reshape(-1, 256, e, e)
            else:
                f_3d = [self.warp(f_s[None], x_s, x_d).reshape(256, e, e)
                        for x_s, x_d, f_s in zip(x_s_batch, x_d_batch, f_s_batch)]
                f_3d = np.stack(f_3d)
            if trace:
                self._t_warp_ms += (time.perf_counter() - t_pre) * 1000
                t_pre = time.perf_counter()
            imgs = self.dec(f_3d)
            if trace:
                self._t_dec_ms += (time.perf_counter() - t_pre) * 1000
                t_pre = time.perf_counter()
            imgs = imgs.permute(0, 2, 3, 1).clamp(0, 1).mul(255) \
                .round().to(torch.uint8)
            # Stage BEFORE the risky boundary and never between: a failure
            # earlier leaves _staged untouched, so emission order (and
            # _take_staged_frame alignment) survives skipped batches.
            self._stage_batch_gpu(imgs)
            if trace:
                self._t_stage_ms += (time.perf_counter() - t_pre) * 1000
                self._stage_n += 1
                if self._stage_n % 25 == 0:
                    logger.info(
                        "[ditto] batches=%d warp=%.1f dec=%.1f stage=%.1f "
                        "(avg ms, gpu thread)", self._stage_n,
                        self._t_warp_ms / self._stage_n,
                        self._t_dec_ms / self._stage_n,
                        self._t_stage_ms / self._stage_n)
                    self._t_warp_ms = self._t_dec_ms = self._t_stage_ms = 0.0
                    self._stage_n = 0

    # ─── emotion ──────────────────────────────────────────────────────────
    def set_emotion(self, value):
        """Session-sticky emotion override; ``None`` clears to the cfg default.

        Returns the parsed value (index / list of indices / None) so callers
        can forward it as a per-utterance tag. Called from the HTTP route
        thread; value may be a name (``"happy"``), an index, or a list of
        indices (upstream averages a list into one soft target). Invalid
        input raises ValueError, which the route surfaces.
        """
        parsed = ditto_emotion.parse_emotion(value)
        # The LMDM consumer runs on the inference thread; a plain attribute
        # write is atomic enough for one writer (HTTP) + one reader (render).
        self._emo_override = parsed
        return parsed

    def current_emotion(self) -> str:
        return ditto_emotion.emotion_name(self._emo_override)

    # ─── audio features ────────────────────────────────────────────────────
    def _make_feat_fn(self):
        _ensure_ditto_home()
        return make_ditto_feat_fn(
            os.path.join(DITTO_WEIGHTS, "aux_models",
                         "hubert_streaming_fix_kv.onnx"),
            device="cuda" if device.type == "cuda" else "cpu")

    # ─── LMDM driving (mirrors upstream _audio2motion_worker) ───────────────
    def _drive_lmdm(self, feats: np.ndarray, emos: Optional[list] = None) -> None:
        seq = self.a2m.seq_frames
        v = self.a2m.valid_clip_len
        self._audio_feat = np.concatenate([self._audio_feat, feats], axis=0)
        # Emotion tags ride in a list parallel to the feature rows, with the
        # same append/trim lifecycle as _audio_feat (guarded identically at
        # the trim below, so the two can never drift out of sync).
        if emos is not None:
            self._emo_rows = self._emo_rows + list(emos)
        else:
            self._emo_rows = self._emo_rows + [None] * feats.shape[0]

        while True:
            window = self._audio_feat[self._local_idx:self._local_idx + seq]
            if window.shape[0] < seq:
                break                                  # wait for more rows
            # One emotion per WINDOW, upstream's granularity: a window spans
            # one utterance boundary at most, so take the mode (most common
            # tag; None counts as a candidate so pure-None stays None).
            w_emos = self._emo_rows[self._local_idx:self._local_idx + seq]
            emo = self._window_emotion(w_emos)
            cond = self.ch(window, self._global_idx, emo=emo)
            if torch.is_tensor(cond):
                cond = cond.cpu().numpy()
            if cond.ndim == 2:
                cond = cond[None]          # LMDM contract: (1, seq, dim)
            t0 = time.perf_counter()
            self._res_kp_seq = self.a2m(cond, self._res_kp_seq)
            self._lmdm_ms += (time.perf_counter() - t0) * 1000
            self._lmdm_passes += 1

            if self._valid_start is None:
                # first online pass: latch d0 only, emit nothing
                self._valid_start = self._res_kp_seq.shape[1] - self.a2m.fuse_length
                self.stitch.d0 = self.a2m.cvt_fmt(self._res_kp_seq[0:1])[0]
            else:
                new = self.a2m.cvt_fmt(
                    self._res_kp_seq[:, self._valid_start:self._valid_start + v])
                for i, xd in enumerate(new):
                    xd["_row"] = self._emit_row + i
                self._x_d_pending.extend(new)
                self._emit_row += v
                self._valid_start += v
                # Drop the oldest entries whose audio has already been played
                # (see _xd_queue_max). Kept newest-first, so the renderer shows
                # motion for the audio being heard NOW instead of a backlog.
                if len(self._x_d_pending) > self._xd_queue_max:
                    del self._x_d_pending[
                        :len(self._x_d_pending) - self._xd_queue_max]

            self._local_idx += v
            self._global_idx += v

            # trim both buffers exactly like upstream (memory + index hygiene)
            L = self._audio_feat.shape[0]
            if L > seq * 2:
                cut = L - seq * 2
                self._audio_feat = self._audio_feat[cut:]
                self._local_idx -= cut
                del self._emo_rows[:cut]
            if self._res_kp_seq is not None and self._res_kp_seq.shape[1] > seq * 2:
                cut = self._res_kp_seq.shape[1] - seq * 2
                self._res_kp_seq = self._res_kp_seq[:, cut:]
                if self._valid_start is not None:
                    self._valid_start -= cut

    def _window_emotion(self, w_emos: list):
        """The window's emotion: its most common tag; sticky override if none.

        Upstream conditions a whole 80-row window on ONE emotion vector, so a
        boundary window (spans a sentence end/start) is resolved as its mode
        rather than chopped. An untagged window falls back to the session
        override so a sticky "happy" keeps applying to untagged speech.
        """
        tagged = [e for e in w_emos if e is not None]
        if tagged:
            emo = max(set(tagged), key=tagged.count)
            if _EMO_TRACE and (self._emo_trace_n or emo != self._last_emo):
                logger.info("[emo] window %d rows=%d tag=%s -> %s",
                            self._global_idx, len(w_emos),
                            ditto_emotion.emotion_name(emo),
                            ditto_emotion.emotion_name(self._emo_override))
                self._last_emo = emo
            return emo
        return self._emo_override

    def _neutral_xd(self):
        """A held-mouth x_d for batches that arrive before the LMDM emits."""
        if self._idle_xd is None:
            if self.stitch.d0 is not None:
                self._idle_xd = self.stitch.d0
            else:
                self._idle_xd = {k: v.copy() for k, v in self.x_s_list[0].items()}
        return self._idle_xd

    # ─── idle lane: give a still image something to do ─────────────────────
    # A single-image artifact has ONE cyclic frame, so the silent branch's
    # `frame_list_cycle[idx]` is a frozen photograph — a dead giveaway. The
    # LMDM cannot produce idle motion here: its state (kp_cond, res_kp_seq,
    # clip_idx, fuse window) belongs to the speech lane and a concurrent pass
    # would corrupt the next utterance. Instead the motion is synthesized
    # directly in x_d space — the same dict MotionStitch consumes as the
    # driving signal — and rendered through warp+decode.
    #
    # Cost model: each idle frame is rendered ONCE and cached, so a full loop
    # costs _IDLE_LOOP renders in total (~50 at 512², a few seconds) and then
    # idle playback is pure CPU. `_IDLE_REUSE` sets how many displayed frames
    # each rendered frame covers, i.e. the idle frame rate.
    #
    # exp and kp are left at the source value: they are 63-dim expression /
    # keypoint offsets whose geometry this lane does not own, and inventing
    # values there renders as face-melt rather than as life.

    def _idle_motion(self, i: int) -> dict:
        """One idle x_d: the source pose plus a breathing sway.

        `i` is the idle loop index; every term is periodic in `_IDLE_LOOP`, so
        the cached loop repeats without a seam.
        """
        x_d = {k: np.array(v, dtype=np.float32, copy=True)
               for k, v in self.x_s_list[0].items()}
        ph = 2.0 * math.pi * (i % _IDLE_LOOP) / float(_IDLE_LOOP)
        # Amplitudes are in the motion space's own units (normalized pose).
        # Small on purpose: enough to read as breathing at 512², small enough
        # that a still head never looks like it is nodding off.
        sway = math.sin(ph)
        drift = math.sin(2.0 * ph)
        for key, amp in (("yaw", 0.010), ("pitch", 0.006),
                         ("roll", 0.004), ("t", 0.004)):
            v = x_d.get(key)
            if v is None or v.size == 0:
                continue
            flat = v.reshape(-1).copy()
            flat[0] += amp * sway
            if key == "t" and flat.size > 1:
                flat[1] += 0.5 * amp * drift
            x_d[key] = flat.reshape(v.shape)
        sc = x_d.get("scale")
        if sc is not None and sc.size:
            # scale is stored as value-1 in this space: a breath nudges it a
            # hair at the swing extremes.
            x_d["scale"] = (sc + 0.0015 * drift).astype(np.float32)
        return x_d

    def _render_idle(self, i: int) -> Optional[np.ndarray]:
        """warp+decode ONE idle frame -> BGR crop, on the GPU worker thread.

        Returns None when no GPU lane is available (CPU-only, or a decode
        object without the batched engine): the caller then keeps the frozen
        source frame, i.e. exactly the old behaviour.
        """
        idx = 0
        try:
            x_s, x_d = self.stitch(self.x_s_list[idx], self._idle_motion(i))
        except Exception:
            logger.exception("[ditto] idle stitch failed")
            return None
        e = self.render_edge

        def job():
            with torch.no_grad():
                if self._warp_pool_staged():
                    f_3d = self.warp.run_batch_dev([(idx, x_s, x_d)])
                    f_3d = torch.stack(f_3d).reshape(-1, 256, e, e)
                else:
                    f_3d = np.stack([
                        self.warp(self.f_s_list[idx][None], x_s, x_d)
                        .reshape(256, e, e)])
                imgs = self.dec(f_3d)
                imgs = imgs.permute(0, 2, 3, 1).clamp(0, 1).mul(255) \
                    .round().to(torch.uint8)
                return imgs[0].cpu().numpy()

        try:
            pipeline = self._pipeline()
            if pipeline is not None:
                out = {}
                pipeline.run_on_gpu(lambda: out.__setitem__("img", job()))
                img = out.get("img")
            else:
                img = job()
        except Exception:
            logger.exception("[ditto] idle render failed; keeping the still frame")
            return None
        if img is None:
            return None
        # The decoder returns RGB; artifact frames are BGR, and paste_back
        # converts an RGB render itself — so hand it RGB and let it convert.
        return np.ascontiguousarray(img)

    def idle_frame(self, idx: int):
        """Rendered idle crop while silent, or None to keep the still frame.

        Only image artifacts take this lane: a video artifact's own clip IS
        its idle. Frames are rendered once each, in loop order, and cached;
        after one full loop idle playback costs no GPU at all.
        """
        if not _IDLE_MOTION or self.n_frames != 1:
            return None
        tick = self._idle_tick
        self._idle_tick += 1
        slot = tick // _IDLE_REUSE
        i = slot % _IDLE_LOOP
        crop = self._idle_cache.get(i)
        if crop is None:
            t0 = time.perf_counter()
            crop = self._render_idle(i)
            if crop is None:
                if getattr(self, "_idle_failed", None) != "render":
                    self._idle_failed = "render"
                    logger.warning("[ditto] idle lane disabled: render returned "
                                   "nothing (still frame kept)")
                return None
            self._idle_cache[i] = crop
            if len(self._idle_cache) == 1:
                logger.info("[ditto] idle lane on: first frame %.0f ms, "
                            "loop %d frames, reuse %d",
                            (time.perf_counter() - t0) * 1000,
                            _IDLE_LOOP, _IDLE_REUSE)
            elif len(self._idle_cache) == _IDLE_LOOP:
                logger.info("[ditto] idle loop complete (%d frames cached); "
                            "idle playback is now GPU-free", _IDLE_LOOP)
        try:
            return self.paste_back_frame(crop, 0)
        except Exception:
            logger.exception("[ditto] idle paste-back failed")
            return None

    def _pick_xd(self):
        """The motion frame to render for THIS batch, by content pointer.

        In drop mode the audio track plays at 50 chunks/s while the renderer
        emits at ~18 fps, so a FIFO consumer always renders motion for audio
        that has already left the track — the measured 1.6 s lag. Each x_d
        entry is tagged with the feature row it came from (1 row = 2 chunks
        = 40 ms of audio), so the entry whose audio is being played NOW is
        the one whose row index is closest to played_chunks/2. Picking it
        instead of the oldest keeps the lips on the audio actually being
        heard; the entries behind it are discarded (their audio is gone).
        Returns the chosen entry with "_row" popped: MotionStitch rebuilds
        the dict, so the tag would be lost by the caller anyway; the caller
        re-reads this queue for the next pick instead.
        """
        if not self._x_d_pending:
            # Starved: the queue ran dry before the batch was filled. Counted
            # so the health log can report it instead of the mouth silently
            # showing the neutral (registration) pose.
            self._neutral_picks = getattr(self, "_neutral_picks", 0) + 1
            return self._neutral_xd()
        played = getattr(self, "_paced_chunk_idx", 0)
        # Each row covers 2 chunks; the ASR's right-context means the emitted
        # row lags the chunk it describes by audio_lead_chunks, so the row
        # describing the chunk currently being played is the one nearest
        # (played - lead) / 2.
        lead = getattr(self.asr, "audio_lead_chunks", 0)
        # LT_XD_TARGET_ADJUST shifts the pick target in rows: >0 picks newer
        # motion (lips ahead), <0 older (lips behind). Calibrate with the lag
        # probe (inspect_lag.py) if a residue remains.
        target = (played - lead) / 2.0 + _XD_TARGET_ADJUST

        # Re-anchor ONLY when the content pointer has advanced since the last
        # pick. The picker is called batch_size times per batch from one tight
        # loop while `played` does not move, and every one of those calls was
        # re-resolving the same target: the first call dropped the stale
        # prefix (best+1 rows for one frame) and the rest took one row each,
        # so a batch consumed ~11 rows while the LMDM emits 10. The queue
        # starved by ~1 row per batch and that frame fell through to the
        # neutral pose — visibly the registration face, mid-sentence
        # (measured: 141 of 160 speaking frames neutral).
        if getattr(self, "_xd_anchor_played", None) != played:
            self._xd_anchor_played = played
            best, best_d = 0, None
            for i, xd in enumerate(self._x_d_pending):
                d = abs(xd["_row"] - target)
                if best_d is None or d < best_d:
                    best, best_d = i, d
            # Strictly older rows describe audio that has already played.
            del self._x_d_pending[:best]
        # Within a batch the pointer stands still, so consecutive frames take
        # consecutive motion rows — the same ordering upstream emits them in.
        chosen = self._x_d_pending.pop(0)
        chosen_row = chosen.pop("_row", None)
        if _XD_TRACE and self._xd_trace_n < 80:
            self._xd_trace_n += 1
            logger.info("[xd] played=%d lead=%d target=%.1f row=%d ahead=%d",
                        played, lead, target, chosen_row,
                        len(self._x_d_pending))
        return chosen

    # ─── batch contract ────────────────────────────────────────────────────
    @torch.no_grad()
    def inference_batch(self, index, audiofeat_batch):
        """Return exactly self.batch_size rendered face crops (RGB->BGR done).

        audiofeat_batch: list of batch_size per-frame HuBERT features, or a
        ``(features, emotion)`` pair when the ASR tags rows (PcmASR does);
        older plain-list callers stay supported.

        With the batched TRT lanes, all batch_size frames go through ONE
        warp chain and ONE decode instead of per-frame chains — 4x fewer GPU
        round-trips per window.
        """
        with torch.no_grad():
            trace = _STAGE_TRACE
            t_pre = t_lmdm = t_stitch = t_warp = t_dec = t_stage = 0.0
            if trace:
                t_pre = time.perf_counter()
            emos = None
            if isinstance(audiofeat_batch, tuple) and len(audiofeat_batch) == 2:
                audiofeat_batch, emos = audiofeat_batch
            feats = np.asarray(audiofeat_batch, dtype=np.float32)
            self._drive_lmdm(feats, emos)
            if trace:
                t_lmdm = time.perf_counter() - t_pre
                t_pre = time.perf_counter()

            x_s_batch, x_d_batch, f_s_batch = [], [], []
            for _ in range(self.batch_size):
                if self._x_d_pending:
                    x_d = self._x_d_pending.pop(0)
                else:
                    x_d = self._neutral_xd()
                idx = _mirror(self._gen_frame_idx, self.n_frames)
                x_s, x_d = self.stitch(self.x_s_list[idx], x_d)
                x_s_batch.append(x_s)
                x_d_batch.append(x_d)
                f_s_batch.append(self.f_s_list[idx].astype(np.float32))
                self._gen_frame_idx += 1
            if trace:
                t_stitch = time.perf_counter() - t_pre
                t_pre = time.perf_counter()

            if hasattr(self.dec, "batch"):
                # Warp output is (C=256, D, H, W) flattened as stored: the graph's
                # reshape is edge-dependent (E=64 -> 512 render, E=32 -> 256).
                e = self.render_edge
                if device.type != "cuda":
                    # No batched engine without CUDA (tests); synchronous path.
                    imgs = self.dec(np.stack(
                        [self.warp(f_s[None], x_s, x_d).reshape(256, e, e)
                         for x_s, x_d, f_s in
                         zip(x_s_batch, x_d_batch, f_s_batch)]))
                    imgs = imgs.permute(0, 2, 3, 1).clamp(0, 1).mul(255) \
                        .round().to(torch.uint8).cpu().numpy()
                    return [np.ascontiguousarray(img) for img in imgs]
                # Batched decode lane: one engine call per window. The per-frame
                # hybrid warp feeds it (numpy in); decode returns (B,3,H,W)
                # torch in sigmoid range. The uint8 conversion happens ON GPU and
                # the host copy is ASYNC into pinned memory (a synchronous .cpu()
                # stalled the inference thread ~27 ms/frame with the GPU idle).
                # Markers flow through the batch contract; paste_back_frame pulls
                # the pixels from the staged ring in strict emission order.
                # (Two-thread lane: never reached — _pipeline() diverts at
                # _submit_batch — but kept as the synchronous reference path.)
                f_3d = [self.warp(f_s[None], x_s, x_d).reshape(256, e, e)
                        for x_s, x_d, f_s in zip(x_s_batch, x_d_batch, f_s_batch)]
                if trace:
                    t_warp = time.perf_counter() - t_pre
                    t_pre = time.perf_counter()
                imgs = self.dec(np.stack(f_3d))
                if trace:
                    t_dec = time.perf_counter() - t_pre
                    t_pre = time.perf_counter()
                imgs = imgs.permute(0, 2, 3, 1).clamp(0, 1).mul(255) \
                    .round().to(torch.uint8)
                self._stage_batch_gpu(imgs)
                if trace:
                    t_stage = time.perf_counter() - t_pre
                    self._t_lmdm_ms += t_lmdm * 1000
                    self._t_stitch_ms += t_stitch * 1000
                    self._t_warp_ms += t_warp * 1000
                    self._t_dec_ms += t_dec * 1000
                    self._t_stage_ms += t_stage * 1000
                    self._stage_n += 1
                    if self._stage_n % 25 == 0:
                        logger.info(
                            "[ditto] batches=%d lmdm=%.1f stitch=%.1f "
                            "warp=%.1f dec=%.1f stage=%.1f (avg ms)",
                            self._stage_n,
                            self._t_lmdm_ms / self._stage_n,
                            self._t_stitch_ms / self._stage_n,
                            self._t_warp_ms / self._stage_n,
                            self._t_dec_ms / self._stage_n,
                            self._t_stage_ms / self._stage_n)
                        self._t_lmdm_ms = self._t_stitch_ms = 0.0
                        self._t_warp_ms = self._t_dec_ms = 0.0
                        self._t_stage_ms = 0.0
                        self._stage_n = 0
                return [_STAGED_FRAME] * self.batch_size

            out = []
            for x_s, x_d, f_s in zip(x_s_batch, x_d_batch, f_s_batch):
                f_3d = self.warp(f_s[None], x_s, x_d)
                img = np.asarray(self.dec(f_3d))
                img = np.clip(img, 0, 255).astype(np.uint8)
                out.append(np.ascontiguousarray(img))
            return out

    # ─── async GPU->host handoff ───────────────────────────────────────────
    def _stage_batch_gpu(self, imgs_uint8_gpu):
        """Kick an async copy of this window's frames into pinned host memory.

        The copy is enqueued on the current stream right behind the decode,
        an event records its completion, and the producer returns without
        waiting. Memory stays bounded by the pipeline itself:
        res_frame_queue (batch_size*4 frames) throttles the inference thread
        long before the staging deque can grow meaningfully.
        """
        host = torch.empty(tuple(imgs_uint8_gpu.shape), dtype=torch.uint8,
                           pin_memory=True)
        host.copy_(imgs_uint8_gpu, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._staged.append({"host": host, "ev": ev, "taken": 0})

    def _take_staged_frame(self) -> np.ndarray:
        """Pop the next frame in emission order, waiting for the copy only at
        the moment the pixels are read."""
        slot = self._staged[0]
        slot["ev"].synchronize()
        img = np.ascontiguousarray(slot["host"].numpy()[slot["taken"]])
        slot["taken"] += 1
        if slot["taken"] >= slot["host"].shape[0]:
            self._staged.popleft()
        return img

    # ─── paste-back (upstream's numpy putback math, restructured) ───────────
    # Same formula as upstream PutBackNumpy (mask*render + (1-mask)*frame),
    # but the per-frame work is cut from two full-frame float warps + ~7
    # full-frame float ops to two bbox-sized uint8/float32 warps + one
    # cv2.blendLinear. Measured on the box (1080x1920 frame): 41.6 -> 10.9
    # ms/frame, max pixel diff 1 (blendLinear rounds instead of truncating).
    # The render crop only lands where the mask is > 0, which is bounded by
    # the mapped 512-crop rectangle, so everything outside the bbox is an
    # untouched copy of the source frame.
    def _paste_geom(self, idx: int, pred_h: int, pred_w: int):
        """(M_crop 2x3, bbox, w, h) for one artifact frame, computed on demand.

        Keyed by idx AND the render-crop size: the same artifact matrix maps
        a 512 render 1:1, but a smaller render covers the SAME crop region at
        fewer pixels — the linear part scales by crop_dsize/pred_w while the
        translation stays (crop origin maps to the same frame position; the
        probe at scratch:semantics_256.py verified the scaled bbox equals the
        512 bbox, and a face pasted through it lands correctly).
        """
        key = (idx, pred_h, pred_w)
        geom = self._paste_geom_cache.get(key)
        if geom is None:
            M = self.M_c2o_list[idx]
            frame = self.frame_list_cycle[idx]
            h, w = frame.shape[:2]
            corners = np.array([[0, 0], [pred_w, 0], [0, pred_h],
                                [pred_w, pred_h]], dtype=np.float32)
            crop_dsize = float(self.marker["render_size"])
            s = crop_dsize / float(pred_w)
            Mc = M[:2].copy()
            Mc[:, :2] *= s
            tx = Mc[0, :2] @ corners.T + Mc[0, 2]
            ty = Mc[1, :2] @ corners.T + Mc[1, 2]
            x0 = max(0, int(np.floor(tx.min())))
            x1 = min(w, int(np.ceil(tx.max())))
            y0 = max(0, int(np.floor(ty.min())))
            y1 = min(h, int(np.ceil(ty.max())))
            # NOTE: M[:2] is a contiguous view of the artifact matrix, so
            # a copy is mandatory before the in-place shift below.
            Mc[0, 2] -= x0
            Mc[1, 2] -= y0
            geom = (Mc, (x0, x1, y0, y1), w, h)
            self._paste_geom_cache[key] = geom
        return geom

    def paste_back_frame(self, pred_frame, idx: int):
        if pred_frame is _STAGED_FRAME:
            pred_frame = self._take_staged_frame()
        frame = self.frame_list_cycle[idx]
        if pred_frame is None:
            return frame.copy()
        if pred_frame.ndim == 2:
            pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_GRAY2BGR)
        else:
            # The decoder returns RGB (upstream contract); the artifact frames
            # are cv2.imread BGR. Upstream's putback converts before warping.
            pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR)
        if self._mask_cache is None or \
                self._mask_cache.shape[:2] != pred_frame.shape[:2]:
            _ensure_ditto_home()
            from core.utils.get_mask import get_mask
            self._mask_cache = get_mask(pred_frame.shape[1],
                                        pred_frame.shape[0], 0.9, 0.9)
        Mc, (x0, x1, y0, y1), w, h = self._paste_geom(
            idx, pred_frame.shape[0], pred_frame.shape[1])
        if x1 <= x0 or y1 <= y0:
            return frame.copy()
        out = frame.copy()
        render = cv2.warpAffine(pred_frame, Mc, (x1 - x0, y1 - y0),
                                flags=cv2.INTER_LINEAR)
        # blendLinear needs CV_32FC1 weights: a (H,W,1) mask warps to 2-D on
        # its own, but collapse an explicit 3-channel one rather than relying
        # on that.
        mask_src = self._mask_cache
        if mask_src.ndim == 3:
            mask_src = mask_src[:, :, 0]
        mask = cv2.warpAffine(mask_src, Mc, (x1 - x0, y1 - y0),
                              flags=cv2.INTER_LINEAR).clip(0, 1)
        out[y0:y1, x0:x1] = cv2.blendLinear(render, out[y0:y1, x0:x1],
                                            mask, 1.0 - mask)
        return out

    def get_avatar_length(self):
        return self.n_frames


def _mirror(index: int, size: int) -> int:
    """Upstream's _mirror_index (same cycling law as utils.image.mirror_index)."""
    if size <= 1:
        return 0
    turn = index // size
    res = index % size
    return res if turn % 2 == 0 else size - res - 1
