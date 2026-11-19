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
import subprocess
import sys
import tempfile

import numpy as np


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
    print(f"[lab] {len(feats)} feature rows ({len(feats)/25:.1f}s of audio)",
          flush=True)

    tmp = tempfile.mkdtemp(prefix="emotion_lab_")
    written = 0
    seq = a2m.seq_frames
    valid = a2m.valid_clip_len
    # The model expects the audio to start with silence: upstream seeds
    # overlap_v2 frames of zeros in front so the first fuse is aligned.
    padded = np.concatenate([np.zeros((seq, feats.shape[1]), np.float32), feats])

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
        for w_start in range(0, padded.shape[0] - seq + 1, valid):
            window = padded[w_start:w_start + seq]
            c = cond(window, w_start, emo=emo)
            if hasattr(c, "cpu"):
                c = c.cpu().numpy()
            if c.ndim == 2:
                c = c[None]
            res_seq = a2m(c, res_seq)
            if w_start == 0:
                # First pass only latches d0 (upstream's own behaviour).
                stitch.d0 = a2m.cvt_fmt(res_seq[0:1])[0]
                continue
            new = a2m.cvt_fmt(res_seq[:, res_seq.shape[1] - valid:])
            for xd in new:
                if len(out_crops) >= len(feats):
                    break
                fidx = len(out_crops) % n
                x_s, x_d = stitch(x_s_lst[fidx], xd)
                f3d = warp(f_s_lst[fidx][None], x_s, x_d).reshape(
                    256, render_edge, render_edge)
                img = dec(np.stack([f3d])) if getattr(dec, "batch", None) else dec(f3d)
                img = np.asarray(img)
                if img.ndim == 4:
                    img = img[0]
                out_crops.append(img)
            if len(out_crops) >= len(feats):
                break

        if not out_crops:
            print(f"[lab] {label}: no frames produced", flush=True)
            continue

        # Burn the label into a strip above each crop and write PNGs.
        import cv2
        h, w = out_crops[0].shape[:2]
        for i, crop in enumerate(out_crops):
            c8 = crop.astype(np.uint8)
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

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25",
           "-pattern_type", "glob", "-i", os.path.join(tmp, "*.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out]
    print("[lab] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"[lab] wrote {args.out} ({written} frames)", flush=True)
    print(f"[lab] source pngs kept in {tmp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
