# tools/enhance — avatar base-video preprocessing

Turns a raw talking-head clip into a clean, standardized vertical (9:16) base
video that LiveTalking's avatar generators (`genavatar.py` for wav2lip /
musetalk) consume unchanged. See `docs/intent/avatar-video-preprocessing.md`
for the confirmed intent and `tasks/plan.md` for the build plan.

## Status (Task 4 — enhancement)

| Stage | Status |
|-------|--------|
| frame read + H.264 encode | implemented |
| RVM person matte + matte sidecar | implemented (Task 3) |
| lighting correction (WB + CLAHE) | implemented (Task 4) |
| GFPGAN face restore | implemented (Task 4) |
| Real-ESRGAN super-res | implemented, **off by default** (Task 4) |
| vertical composition on flat background | implemented |

The tool removes the background (RVM), enhances the person (lighting + GFPGAN
face restore), composites onto a flat-color 9:16 canvas, and can write a matte
sidecar.

Upscale decision: super-res is **off by default** (`--upscale 1`). Real-ESRGAN
4x from a ~1080p source is minutes-per-frame on a MacBook and the avatar
pipeline downsizes faces to 96/256 anyway, so it's wasted compute for typical
sources. Use `--upscale 4` only for genuinely low-res clips (documented slow).
Face restore (GFPGAN) runs at native resolution in the default path.

Weights (download once):
```
/opt/miniconda3/envs/livetalking/bin/python scripts/download_enhance_weights.py
```
- models/RealESRGAN_x4plus.pth (67 MB) — used only when --upscale > 1
- models/GFPGANv1.4.pth (348 MB) — face restore
- gfpgan/weights/* (auto-downloaded by facexlib on first run: retinaface
  detection 104 MB + face parsing 81 MB)

Measured on MPS (macOS): default path ~2.3s/frame enhance + ~0.1s/frame matting;
5 frames end-to-end in ~14s.

## Usage

```bash
PY=/opt/miniconda3/envs/livetalking/bin/python

# one command: enhance + create an avatar (wav2lip by default, MODEL=musetalk for musetalk)
./tools/enhance/make_avatar.sh raw.mov my_avatar
MODEL=musetalk ./tools/enhance/make_avatar.sh raw.mov my_avatar

# or step by step
# 1) enhance: raw clip -> 1080x1920 flat-background mp4 (+ matte/person sidecars)
"$PY" -m tools.enhance.cli -i raw.mov -o clean.mp4 \
    --matte-out /tmp/matte --person-out /tmp/person

# 2) create the avatar from the enhanced clip
"$PY" -m avatars.wav2lip.genavatar --video_path clean.mp4 --avatar_id my_avatar
"$PY" -m avatars.musetalk.genavatar --file clean.mp4 --avatar_id my_avatar

# 3) swap the background later — no re-matting, no re-enhancing (seconds)
"$PY" -m tools.enhance.cli -i raw.mov -o swapped.mp4 \
    --background '#1E50FF' --recompose-from /tmp/matte --person-out /tmp/person

# custom canvas / color / framing
"$PY" -m tools.enhance.cli -i raw.mov -o clean.mp4 \
    --width 1080 --height 1920 --background '#1E1E1E' --fit contain

# fast smoke test (first 30 frames)
"$PY" -m tools.enhance.cli -i raw.mov -o /tmp/clean.mp4 --max-frames 30
```

Then feed the result to avatar creation as usual:

```bash
"$PY" -m avatars.musetalk.genavatar --file clean.mp4 --avatar_id my_avatar
```

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-i/--input` | — | input video path |
| `-o/--output` | — | output mp4 path |
| `--width` / `--height` | 1080 / 1920 | vertical canvas size |
| `--background` | `#1E1E1E` | flat bg color (`#RRGGBB`, `#RGB`, or `r,g,b`) |
| `--fit` | `contain` | `contain` = letterbox; `cover` = fill + center-crop |
| `--strength` | 0.5 | enhancement strength 0..1 (blend toward full correction) |
| `--upscale` | 1 | Real-ESRGAN factor (1 = native-res face restore, fast; 2/4 for low-res sources, slow) |
| `--no-face-restore` | off | skip GFPGAN (lighting/SR only) |
| `--recompose-from` | — | background swap from a saved matte dir |
| `--max-frames` | all | cap frame count (fast tests) |
| `--matte-out` | — | person-matte sidecar dir (grayscale PNGs + meta.json) |
| `--person-out` | — | person color layer sidecar (BGR PNGs; needed for a lossless bg swap) |
| `--recompose-from` | — | re-compose a saved matte onto a new `--background` (no re-matting) |
| `--quiet` | off | suppress progress |

Exit codes: `0` success, `2` bad input/argument.

## Tests

```bash
"$PY" -m tests.test_enhance_pipeline   # pure-function unit tests, no models
```

## Design notes

- Encoding goes through **ffmpeg → H.264/yuv420p MP4** so the output is broadly
  playable and ingestible by `video2imgs`. Falls back to `cv2.VideoWriter`.
- The person/background split is kept as a separate `alpha` layer all the way
  through `compose_vertical`, so a later background swap (or transparent output)
  is a compositor change, not a re-matting pass.
- `--fit contain` never crops the person (safe default); `cover` fills the frame.
