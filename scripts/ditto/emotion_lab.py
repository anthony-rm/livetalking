#!/usr/bin/env python3
"""Emotion lab — render a grid of Ditto emotions (and soft mixes) to one mp4.

RUNS ON THE BOX (it needs the Ditto weights, TensorRT engines and a GPU); the
output is a single video you download and watch, so no judgement call has to be
made blind.

Why a mix grid rather than just the 8 labels: the conditioning input is an
8-dim vector, not 8 discrete states (`ConditionHandler` concatenates
`aud_feat[1024] + emo[8] + ...` and averages a LIST into one soft target). Any
convex combination is therefore legal input, and blends are where the useful
expressiveness this model has beyond its 8 labels lives: "excited" is a
happy/surprise mix, "content" is a happy/neutral mix.

Usage (box):
    LT_PYTHON=$HOME/miniconda3/envs/livetalking/bin/python
    $LT_PYTHON scripts/ditto/emotion_lab.py \
        --avatar <avatar_id> --audio example/audio.wav --out /tmp/emotions.mp4

Each line of the video is one emotion setting, one after another, with the
label burned into the frame. Segments are separated by a short gap so a seam
is unmistakable.

The render path is deliberately the OFFLINE one (`Audio2Motion` +
`MotionStitch` per frame): this is a quality study, not a throughput test, and
the offline window (overlap_v2=10, 50 steps) is the model's best-effort output.
"""
from __future__ import annotations

import argparse
import os

import sys
import tempfile

import numpy as np

# Run as a script from anywhere: the repo root must be importable so
# `avatars.ditto.*` resolves (the app always runs with it as cwd).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)


def parse_pairs(spec: str):
    """`happy`, `happy+surprise`, ... -> [["happy"], ["happy", "surprise"], ...]."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append([p.strip() for p in chunk.split("+") if p.strip()])
    return out


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    import soundfile as sf
    data, got = sf.read(path, dtype="float32", always_2d=True)
    if got != sr:
        import resampy
        data = resampy.resample(data[:, 0], got, sr)[:, None]
    return data[:, 0].astype(np.float32)


DEFAULT_GRID = ("neutral,happy,happy+surprise,happy+neutral,"
                "sad,sad+neutral,surprise,angry,angry+disgust,"
                "fear,disgust,contempt")


def encode_mp4(frames_dir: str, out_path: str, fps: int = 25) -> None:
    """Encode the PNG sequence with PyAV (the box has no ffmpeg binary).

    PyAV is already a hard dependency of this app (the TTS and WebRTC paths use
    it), so reusing it here keeps the lab runnable on the box as-is.
    """
    import av
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not files:
        raise RuntimeError(f"no png frames in {frames_dir}")
    import cv2
    first = cv2.imread(os.path.join(frames_dir, files[0]))
    h, w = first.shape[:2]
    container = av.open(out_path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    for name in files:
        bgr = cv2.imread(os.path.join(frames_dir, name))
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def render_motion(stitch, warp, dec, x_s_lst, f_s_lst, n, render_edge,
                  motion, n_frames):
    """Render a (T, 265) motion array to CHW crops via stitch+warp+decode.

    Split out from `render_setting` so a caller can transform the MOTION before
    it is rendered — which is what pose exaggeration needs.
    """
    out = []
    for i in range(min(n_frames, motion.shape[0])):
        fidx = i % n
        xd = _arr2dic(motion[i])
        x_s, x_d = stitch(x_s_lst[fidx], xd)
        f3d = warp(f_s_lst[fidx][None], x_s, x_d).reshape(
            256, render_edge, render_edge)
        img = dec(np.stack([f3d])) if getattr(dec, "batch", None) else dec(f3d)
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 4:
            img = img[0]
        out.append(img)
    return out


#: Channel layout of the 265-dim motion vector (upstream `_cvt_LP_motion_info`,
#: kp dropped). Emotion is measured to live almost entirely in the POSE groups:
#: on the box, |label - neutral| averaged yaw 0.084 / roll 0.042 / pitch 0.038
#: against exp 0.0016. So "exaggerate the emotion" means "exaggerate the pose
#: delta", and touching `exp` would invent a smile the model never offered.
MOTION_GROUPS = (("scale", 1), ("pitch", 66), ("yaw", 66), ("roll", 66),
                 ("t", 3), ("exp", 63))


def _motion_spans():
    s = 0
    for name, size in MOTION_GROUPS:
        yield name, s, s + size
        s += size


def _arr2dic(vec):
    """(265,) -> the per-key dict MotionStitch consumes (upstream's reshape)."""
    dic = {}
    for name, s, e in _motion_spans():
        v = vec[s:e].reshape(1, -1)
        dic[name] = (v + 1) if name == "scale" else v
    return dic


def exaggerate_pose(motion, neutral, gain, groups=("pitch", "yaw", "roll", "t")):
    """Amplify the label's pose deviation from neutral, leaving `exp` alone.

    `gain` 1 returns `motion` untouched. The delta is added back to NEUTRAL, so
    the result is always a pose the model could plausibly have produced — this
    is a stronger reading of the same label, not a new one.
    """
    if gain == 1.0 or neutral is None:
        return motion
    out = motion.copy()
    for name, s, e in _motion_spans():
        if name in groups:
            out[:, s:e] = neutral[:, s:e] + gain * (motion[:, s:e]
                                                    - neutral[:, s:e])
    return out


def reset_sampler(a2m, cond, dkw):
    """Return `a2m` to its post-`setup` state before a new setting is sampled.

    `Audio2Motion` carries state ACROSS calls that makes renders order-dependent:
    `clip_idx` drives the `fix_kp_cond` reset cadence, and the caller passes the
    accumulated `res_kp_seq` back in. Sampling setting 2 after setting 1 in one
    process therefore does not reproduce what setting 2 does in its own process.

    That is not a theoretical concern: a neutral render (where pose gain is a
    mathematical no-op) differed by 7.4/255 between two sweep runs, which
    invalidates any per-label comparison drawn from one process. Resetting here
    makes a single-process sweep equivalent to N separate runs.
    """
    a2m.setup(cond.x_s_info_0,
              fix_kp_cond=dkw["fix_kp_cond"],
              fix_kp_cond_dim=dkw["fix_kp_cond_dim"],
              v_min_max_for_clip=dkw["v_min_max_for_clip"],
              smo_k_d=dkw["smo_k_d"], overlap_v2=10,
              sampling_timesteps=getattr(a2m, "sampling_timesteps", 50)
              or 50, online_mode=False)


def sample_motion(cond, a2m, stitch, padded, seq, valid, n_frames, emo, seed):
    """Run the LMDM over `padded` and return the (T, 265) motion array.

    This is the half of the lab that costs GPU time; splitting it from the
    render is what lets a caller transform motion (pose exaggeration) before a
    single frame is drawn.

    `a2m` returns the ACCUMULATED motion history, not just this pass's frames:
    after pass k the result is (1, 80 + k*valid, 265). The new frames are the
    last `valid` columns, and taking them by a pointer (the online lane's law)
    is what keeps a long history from being re-read as if it were the current
    window.
    """
    import torch
    if seed is not None:
        torch.manual_seed(seed)

    res_seq = None
    valid_start = None
    collected = []
    stitch.d0 = None
    for w_start in range(0, padded.shape[0] - seq + 1, valid):
        c = cond(padded[w_start:w_start + seq], w_start, emo=emo)
        if hasattr(c, "cpu"):
            c = c.cpu().numpy()
        if c.ndim == 2:
            c = c[None]
        res_seq = a2m(c, res_seq)
        if valid_start is None:
            # First pass latches d0 only (upstream's own behaviour).
            valid_start = res_seq.shape[1] - a2m.fuse_length
            stitch.d0 = a2m.cvt_fmt(res_seq[0:1])[0]
            continue
        collected.append(res_seq[:, valid_start:valid_start + valid])
        valid_start += valid
        if sum(x.shape[1] for x in collected) >= n_frames:
            break
    if not collected:
        return None
    return np.concatenate(collected, axis=1)[0][:n_frames]


def render_setting(cond, a2m, stitch, warp, dec, x_s_lst, f_s_lst, n,
                   render_edge, padded, seq, valid, n_frames, emo, seed):
    """Sample then render one setting; kept for callers that want it in one go."""
    motion = sample_motion(cond, a2m, stitch, padded, seq, valid, n_frames,
                           emo, seed)
    if motion is None:
        return []
    return render_motion(stitch, warp, dec, x_s_lst, f_s_lst, n, render_edge,
                         motion, n_frames)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--avatar", required=True, help="avatar id under data/avatars/")
    ap.add_argument("--audio", required=True, help="16 kHz mono wav to speak")
    ap.add_argument("--out", default="emotions.mp4")
    ap.add_argument("--grid", default=DEFAULT_GRID,
                    help="comma-separated settings; a+b averages a and b")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="seconds of audio per setting")
    ap.add_argument("--render-size", type=int, default=512, choices=(256, 512))
    ap.add_argument("--steps", type=int, default=50,
                    help="LMDM sampling steps (offline default 50)")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="emotion strength: 1 = upstream, 2 = twice the "
                         "distance from neutral, 0 = neutral")
    ap.add_argument("--guidance", type=float, default=None,
                    help="override LMDM guidance_weight (upstream 2). "
                         "Applies to the WHOLE render, not per emotion.")
    ap.add_argument("--pose-gain", type=float, default=1.0,
                    help="exaggerate the label's POSE deviation from neutral "
                         "(pitch/yaw/roll/t). 1 = as the model produced it. "
                         "This is the knob that actually changes how strong an "
                         "emotion reads")
    ap.add_argument("--seed", type=int, default=1234,
                    help="sampling-noise seed; fixed by default so two "
                         "settings differ by emotion, not by noise")
    args = ap.parse_args()

    os.environ.setdefault("DITTO_RENDER_SIZE", str(args.render_size))
    # The lab renders offline on purpose; the online kwargs are for serving.
    os.environ["LT_LMDM_ASYNC"] = "0"

    from avatars.ditto import emotion as de
    from avatars.ditto.artifact import validate

    avatar_dir = os.path.join(".", "data", "avatars", args.avatar)
    marker = validate(avatar_dir)

    import avatars.ditto_avatar as da
    if not os.path.isdir(os.path.expanduser(da.DITTO_HOME)):
        print(f"error: DITTO_HOME not found: {da.DITTO_HOME}", file=sys.stderr)
        return 2

    settings = parse_pairs(args.grid)
    if not settings:
        print("error: empty --grid", file=sys.stderr)
        return 2

    print(f"[lab] avatar={args.avatar} frames={marker['frame_count']} "
          f"render={args.render_size} settings={len(settings)}", flush=True)

    # Model + artifact, the same loaders the server uses.
    model = da.load_model()
    avatar = da.load_avatar(args.avatar)
    (mk, frames, x_s_lst, f_s_lst, M_c2o_lst, eye_open, eye_ball) = avatar
    n = len(frames)
    dkw = model["default_kwargs"]

    edge = int(mk["render_size"]) // 8
    warp_edge = getattr(model["warp"], "feature_edge", None) or edge
    if warp_edge < edge:
        f_s_lst = da._shrink_f_s_list(f_s_lst, warp_edge)
    render_edge = warp_edge

    cond = model["cond"]
    cond.setup({"x_s_info_lst": x_s_lst,
                "sc": x_s_lst[0]["kp"].reshape(-1),
                "eye_open_lst": ([eye_open[i][None] for i in range(n)]
                                 if eye_open is not None
                                 else [np.zeros((1, 2), np.float32)] * n),
                "eye_ball_lst": ([eye_ball[i][None] for i in range(n)]
                                 if eye_ball is not None
                                 else [np.zeros((1, 6), np.float32)] * n)},
               dkw["emo"], eye_f0_mode=dkw["eye_f0_mode"])

    a2m = model["a2m"]
    a2m.setup(cond.x_s_info_0,
              fix_kp_cond=dkw["fix_kp_cond"], fix_kp_cond_dim=dkw["fix_kp_cond_dim"],
              v_min_max_for_clip=dkw["v_min_max_for_clip"], smo_k_d=dkw["smo_k_d"],
              overlap_v2=10, sampling_timesteps=args.steps, online_mode=False)
    stitch = model["stitch"]
    stitch.setup(N_d=-1, use_d_keys=dkw["use_d_keys"], relative_d=dkw["relative_d"],
                 drive_eye=dkw["drive_eye"], delta_eye_arr=dkw["delta_eye_arr"],
                 delta_eye_open_n=dkw["delta_eye_open_n"],
                 fade_out_keys=dkw["fade_out_keys"], fade_type=dkw["fade_type"],
                 flag_stitching=dkw["flag_stitching"],
                 is_image_flag=(n == 1), x_s_info=x_s_lst[0], d0=None,
                 ch_info=None, overall_ctrl_info=dkw["overall_ctrl_info"])
    warp, dec = model["warp"], model["dec"]

    if args.guidance is not None:
        # `MotionDecoder.guided_forward` extrapolates away from the
        # unconditioned prediction: out = unc + (cond - unc) * weight, with
        # weight=2 upstream. Raising it is the standard way to make a
        # conditional signal assert itself harder.
        #
        # It is only reachable in the PYTORCH build. This box loads TensorRT
        # (`model_type == 'tensorrt'`), where the guidance weight is frozen
        # into the compiled engine: there is no attribute to set, and quietly
        # ignoring --guidance would report a sweep that never happened.
        inner = getattr(a2m.lmdm, "model", None)
        mtype = getattr(a2m.lmdm, "model_type", "?")
        if inner is None or not hasattr(inner, "guidance_weight"):
            print(f"error: this LMDM is '{mtype}', which exposes no "
                  f"guidance_weight (TensorRT bakes it into the engine). "
                  f"Run the pytorch build, or skip --guidance.",
                  file=sys.stderr)
            return 2
        inner.guidance_weight = float(args.guidance)
        print(f"[lab] guidance_weight={inner.guidance_weight} "
              f"(upstream 2)", flush=True)

    # Features once: every setting speaks the SAME audio, which is what makes
    # a grid comparable — only the emotion vector changes.
    sr = 16000
    pcm = load_audio(args.audio, sr)
    feat_fn = da.make_ditto_feat_fn(
        os.path.join(da.DITTO_WEIGHTS, "aux_models", "hubert_streaming_fix_kv.onnx"),
        device="cuda")
    window, stride = 6480, 3200
    rows = []
    buf = np.concatenate([np.zeros(window - stride, np.float32), pcm])
    for start in range(0, max(0, len(buf) - window) + 1, stride):
        chunk = buf[start:start + window]
        if chunk.shape[0] < window:
            chunk = np.pad(chunk, (0, window - chunk.shape[0]))
        rows.append(feat_fn(chunk))
    feats = np.concatenate(rows, axis=0) if rows else np.zeros((1, 1024), np.float32)
    n_frames = int(min(len(feats), args.seconds * 25))
    feats = feats[:max(n_frames, 1)]
    n_frames = len(feats)
    print(f"[lab] {len(feats)} feature rows ({len(feats)/25:.1f}s of audio)",
          flush=True)

    tmp = tempfile.mkdtemp(prefix="emotion_lab_")
    written = 0
    seq = a2m.seq_frames
    valid = a2m.valid_clip_len
    # Two delays the loop must respect, both learned from a run that rendered
    # NOTHING:
    #   * upstream seeds `seq` frames of silence in front so the first fuse is
    #     aligned, and
    #   * the FIRST pass only latches d0, so at least one more pass must exist.
    # A short clip (30 frames) is shorter than one window (80), leaving a
    # single pass and therefore zero output. Pad the tail with silence until
    # two full passes fit, and render only the first `n_frames` results.
    body = feats
    need = seq + 2 * valid
    if body.shape[0] < need:
        body = np.concatenate(
            [body, np.zeros((need - body.shape[0], body.shape[1]), np.float32)])
    padded = np.concatenate([np.zeros((seq, body.shape[1]), np.float32), body])

    # Motion is produced once per setting and then rendered, rather than
    # rendered inline: pose exaggeration transforms the motion array before any
    # frame is drawn, so a `--pose-gain` run and a plain run share one sampler.
    neutral_motion = None
    if args.pose_gain != 1.0:
        neutral_label = de.emotion_name(de.EMOTION_INDEX["neutral"])
        print(f"[lab] sampling neutral motion (pose reference for "
              f"--pose-gain {args.pose_gain:g})", flush=True)
        reset_sampler(a2m, cond, dkw)
        neutral_motion = sample_motion(
            cond, a2m, stitch, padded, seq, valid, n_frames,
            de.emotion_vector(de.EMOTION_INDEX["neutral"], gain=args.gain),
            args.seed)
    for si, labels in enumerate(settings):
        idxs = [de.EMOTION_INDEX[l.lower()] for l in labels
                if l.lower() in de.EMOTION_INDEX]
        if not idxs:
            print(f"[lab] skip unknown setting {labels}", flush=True)
            continue
        base = idxs[0] if len(idxs) == 1 else idxs
        label = "+".join(labels)
        # The conditioning vector, not the label: `emotion_vector` is where
        # gain is applied, so the render and the API agree on what "gain" means.
        emo = de.emotion_vector(base, gain=args.gain)
        print(f"[lab] {si+1}/{len(settings)} {label} -> "
              f"{de.emotion_name(base)} vec_gain={args.gain:g} "
              f"pose_gain={args.pose_gain:g}", flush=True)

        reset_sampler(a2m, cond, dkw)
        motion = sample_motion(cond, a2m, stitch, padded, seq, valid,
                               n_frames, emo, args.seed)
        if motion is None:
            print(f"[lab] {label}: no motion produced", flush=True)
            continue
        motion = exaggerate_pose(motion, neutral_motion, args.pose_gain)
        out_crops = render_motion(stitch, warp, dec, x_s_lst, f_s_lst, n,
                                  render_edge, motion, n_frames)

        if not out_crops:
            print(f"[lab] {label}: no frames produced", flush=True)
            continue

        # Burn the label into a strip above each crop and write PNGs.
        import cv2
        h, w = out_crops[0].shape[1:] if out_crops[0].shape[0] == 3 \
            else out_crops[0].shape[:2]
        for i, crop in enumerate(out_crops):
            c8 = crop
            # The decoder's contract is CHW (3, H, W); cv2 wants HWC BGR.
            if c8.ndim == 3 and c8.shape[0] == 3:
                c8 = np.transpose(c8, (1, 2, 0))
            if c8.dtype != np.uint8:
                c8 = np.clip(c8 * 255.0 if c8.max() <= 1.0 else c8,
                             0, 255).astype(np.uint8)
            if c8.ndim == 3 and c8.shape[2] == 3:
                c8 = cv2.cvtColor(c8, cv2.COLOR_RGB2BGR)
            canvas = np.zeros((h + 28, w, 3), np.uint8)
            canvas[:28] = (28, 28, 28)
            canvas[28:] = c8
            cv2.putText(canvas, f"{label}  pose={args.pose_gain:g}",
                        (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imwrite(os.path.join(tmp, f"{si:03d}_{i:05d}.png"), canvas)
            written += 1
        # gap frames so seams are obvious
        for _ in range(6):
            cv2.imwrite(os.path.join(tmp, f"{si:03d}_gap{_:02d}.png"),
                        np.zeros((h + 28, w, 3), np.uint8))

    if not written:
        print("error: nothing rendered", file=sys.stderr)
        return 1

    encode_mp4(tmp, args.out)
    print(f"[lab] wrote {args.out} ({written} frames)", flush=True)
    print(f"[lab] source pngs kept in {tmp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
