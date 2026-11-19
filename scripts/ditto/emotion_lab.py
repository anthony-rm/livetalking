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

    for si, labels in enumerate(settings):
        idxs = [de.EMOTION_INDEX[l.lower()] for l in labels
                if l.lower() in de.EMOTION_INDEX]
        if not idxs:
            print(f"[lab] skip unknown setting {labels}", flush=True)
            continue
        emo = idxs[0] if len(idxs) == 1 else idxs
        label = "+".join(labels)
        print(f"[lab] {si+1}/{len(settings)} {label} -> emo={emo}", flush=True)

        out_crops = []
        res_seq = None      # fresh per setting: no carry-over between settings
        # `a2m` returns the ACCUMULATED motion history, not just this pass's
        # frames: after pass k the result is (1, 80 + k*valid, 265). The new
        # frames are the last `valid` columns, and taking them by a pointer
        # (the online lane's law) is what keeps a long history from being
        # re-read as if it were the current window.
        valid_start = None
        for w_start in range(0, padded.shape[0] - seq + 1, valid):
            window = padded[w_start:w_start + seq]
            c = cond(window, w_start, emo=emo)
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
            new = a2m.cvt_fmt(res_seq[:, valid_start:valid_start + valid])
            valid_start += valid
            for xd in new:
                if len(out_crops) >= n_frames:
                    break
                fidx = len(out_crops) % n
                x_s, x_d = stitch(x_s_lst[fidx], xd)
                f3d = warp(f_s_lst[fidx][None], x_s, x_d).reshape(
                    256, render_edge, render_edge)
                img = dec(np.stack([f3d])) if getattr(dec, "batch", None) else dec(f3d)
                if hasattr(img, "detach"):
                    img = img.detach().cpu().numpy()
                img = np.asarray(img)
                if img.ndim == 4:
                    img = img[0]
                out_crops.append(img)
            if len(out_crops) >= n_frames:
                break

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
            cv2.putText(canvas, f"{label}  [{emo}]", (6, 20),
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
