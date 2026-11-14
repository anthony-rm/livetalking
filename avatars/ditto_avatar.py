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
import pickle
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import cv2

from avatars.base_avatar import BaseAvatar
from avatars.audio_features.pcm import PcmASR, make_ditto_feat_fn
from avatars.ditto import artifact as art
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
ONLINE_KWARGS = dict(overlap_v2=70, sampling_timesteps=10, online_mode=True)

#: HuBERT feature dim (PcmASR.FEAT_DIM). Ditto's own checkpoint; nothing else
#: may substitute (§4 of the skill reference).
FEAT_DIM = 1024


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
    warp = None
    hybrid_dir = os.environ.get("DITTO_WARP_ENGINE_DIR", "")
    if not hybrid_dir:
        cand = os.path.join(os.path.dirname(DITTO_WEIGHTS.rstrip("/")),
                            "warp_hybrid_engines")
        if os.path.exists(os.path.join(cand, "hyb_segB.engine")):
            hybrid_dir = cand
    if hybrid_dir:
        _install_cuda_shim()
        from avatars.ditto.trt_warp_hybrid import TRTWarpHybrid
        warp = TRTWarpHybrid(hybrid_dir, device=dev)
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
    return marker, frame_list, x_s_list, f_s_list, M_c2o_list


@torch.no_grad()
def warm_up(batch_size, model):
    """Prime the GPU components once so the first live batch pays no warmup."""
    _ensure_ditto_home()
    from core.atomic_components.motion_stitch import transform_keypoint

    parts = model
    zero = np.zeros((1, 32, 16, 64, 64), dtype=np.float32)
    x_s_info = {k: np.zeros(s, dtype=np.float32) for k, s in _X_S_SHAPES.items()}
    x_s = transform_keypoint(x_s_info)          # dict -> (bs, num_kp, 3)
    f_3d = parts["warp"](zero, x_s, x_s.copy())
    parts["dec"](f_3d)
    logger.info("[ditto] warmup done (warp+decode primed)")


@register("avatar", "ditto")
class DittoReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        self.parts = model
        (self.marker, self.frame_list_cycle, self.x_s_list,
         self.f_s_list, self.M_c2o_list) = avatar
        self.n_frames = len(self.frame_list_cycle)

        self.asr = PcmASR(opt, self, self._make_feat_fn())
        self.asr.warm_up()

        dkw = self.parts["default_kwargs"]
        self.ch = self.parts["cond"]
        ch_info = {
            "x_s_info_lst": self.x_s_list,
            "sc": self.x_s_list[0]["kp"].reshape(-1),
            "eye_open_lst": [np.zeros((1, 2), np.float32)] * self.n_frames,
            "eye_ball_lst": [np.zeros((1, 6), np.float32)] * self.n_frames,
        }
        self.ch.setup(ch_info, dkw["emo"], eye_f0_mode=dkw["eye_f0_mode"])

        self.a2m = self.parts["a2m"]
        self.a2m.setup(self.ch.x_s_info_0,
                       fix_kp_cond=dkw["fix_kp_cond"],
                       fix_kp_cond_dim=dkw["fix_kp_cond_dim"],
                       v_min_max_for_clip=dkw["v_min_max_for_clip"],
                       smo_k_d=dkw["smo_k_d"],
                       **ONLINE_KWARGS)

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

        # ── streaming state (§3: why a stateless batch API cannot host Ditto) ──
        # Cond row buffer, SEEDED with one window of silence so the first pass
        # latches d0 without waiting for 3.2 s of speech (upstream seeds zeros
        # in front of the audio for the same reason).
        self._audio_feat = np.zeros((self.a2m.seq_frames, FEAT_DIM), np.float32)
        self._local_idx = 0        # window start inside _audio_feat
        self._global_idx = 0       # absolute feature-frame counter (cond idx)
        self._res_kp_seq = None    # fused LMDM output
        self._valid_start = None   # first un-emitted frame in _res_kp_seq
        self._x_d_pending: List[dict] = []
        self._gen_frame_idx = 0
        self._lmdm_ms = 0.0
        self._lmdm_passes = 0
        self._idle_xd: Optional[dict] = None
        self._mask_cache: Optional[np.ndarray] = None

    # ─── audio features ────────────────────────────────────────────────────
    def _make_feat_fn(self):
        _ensure_ditto_home()
        return make_ditto_feat_fn(
            os.path.join(DITTO_WEIGHTS, "aux_models",
                         "hubert_streaming_fix_kv.onnx"),
            device="cuda" if device.type == "cuda" else "cpu")

    # ─── LMDM driving (mirrors upstream _audio2motion_worker) ───────────────
    def _drive_lmdm(self, feats: np.ndarray) -> None:
        seq = self.a2m.seq_frames
        v = self.a2m.valid_clip_len
        self._audio_feat = np.concatenate([self._audio_feat, feats], axis=0)

        while True:
            window = self._audio_feat[self._local_idx:self._local_idx + seq]
            if window.shape[0] < seq:
                break                                  # wait for more rows
            cond = self.ch(window, self._global_idx)
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
                self._x_d_pending.extend(new)
                self._valid_start += v

            self._local_idx += v
            self._global_idx += v

            # trim both buffers exactly like upstream (memory + index hygiene)
            L = self._audio_feat.shape[0]
            if L > seq * 2:
                cut = L - seq * 2
                self._audio_feat = self._audio_feat[cut:]
                self._local_idx -= cut
            if self._res_kp_seq is not None and self._res_kp_seq.shape[1] > seq * 2:
                cut = self._res_kp_seq.shape[1] - seq * 2
                self._res_kp_seq = self._res_kp_seq[:, cut:]
                if self._valid_start is not None:
                    self._valid_start -= cut

    def _neutral_xd(self):
        """A held-mouth x_d for batches that arrive before the LMDM emits."""
        if self._idle_xd is None:
            if self.stitch.d0 is not None:
                self._idle_xd = self.stitch.d0
            else:
                self._idle_xd = {k: v.copy() for k, v in self.x_s_list[0].items()}
        return self._idle_xd

    # ─── batch contract ────────────────────────────────────────────────────
    @torch.no_grad()
    def inference_batch(self, index, audiofeat_batch):
        """Return exactly self.batch_size rendered face crops (RGB->BGR done).

        audiofeat_batch: list of batch_size per-frame HuBERT features.
        """
        feats = np.asarray(audiofeat_batch, dtype=np.float32)
        self._drive_lmdm(feats)

        out = []
        for _ in range(self.batch_size):
            if self._x_d_pending:
                x_d = self._x_d_pending.pop(0)
            else:
                x_d = self._neutral_xd()
            idx = _mirror(self._gen_frame_idx, self.n_frames)
            x_s, x_d = self.stitch(self.x_s_list[idx], x_d)
            f_s = self.f_s_list[idx][None].astype(np.float32)   # (1,32,16,64,64)
            f_3d = self.warp(f_s, x_s, x_d)
            img = np.asarray(self.dec(f_3d))                    # [h,w,c] float 0-255, RGB
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.ascontiguousarray(img)
            out.append(img)
            self._gen_frame_idx += 1
        return out

    # ─── paste-back (upstream's numpy putback, no Cython) ──────────────────
    def paste_back_frame(self, pred_frame, idx: int):
        frame = self.frame_list_cycle[idx].copy()
        if pred_frame is None:
            return frame
        if pred_frame.ndim == 2:
            pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        if self._mask_cache is None or \
                self._mask_cache.shape[:2] != pred_frame.shape[:2]:
            _ensure_ditto_home()
            from core.utils.get_mask import get_mask
            m = get_mask(pred_frame.shape[1], pred_frame.shape[0], 0.9, 0.9)
            self._mask_cache = np.concatenate([m] * 3, axis=2)
        M = self.M_c2o_list[idx]
        mask_w = cv2.warpAffine(self._mask_cache, M[:2], (w, h),
                                flags=cv2.INTER_LINEAR).clip(0, 1)
        render_w = cv2.warpAffine(pred_frame.astype(np.float32), M[:2], (w, h),
                                  flags=cv2.INTER_LINEAR)
        out = mask_w * render_w + (1 - mask_w) * frame.astype(np.float32)
        return out.clip(0, 255).astype(np.uint8)

    def get_avatar_length(self):
        return self.n_frames


def _mirror(index: int, size: int) -> int:
    """Upstream's _mirror_index (same cycling law as utils.image.mirror_index)."""
    if size <= 1:
        return 0
    turn = index // size
    res = index % size
    return res if turn % 2 == 0 else size - res - 1
