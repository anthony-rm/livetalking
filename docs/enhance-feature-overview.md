# Video enhancer
---

## 1. What it does

Takes a raw talking-head clip and
produces a clean, standardized clean base video for the avatar generation:

1. the person is cut out of the background (matte),
2. the lighting is corrected,
3. the face is restored,
4. person + matte are recomposited on a flat-color vertical canvas,
5. the result is encoded to H.264.

That output is then fed unchanged to any avatar generator (wav2lip,
musetalk) — no change to `genavatar.py`, no change to any backend.

---

## 2. Pipeline

```
 raw clip
   │
   ├── open_reader()          cv2.VideoCapture + fps guard (nan/0 → 30)
   │
   ├── compute_alpha()        RVM mobilenetv3 → alpha uint8 HxW (255 = person)
   │                          recurrent: 4 hidden states carried frame → frame,
   │                          reset at the start of each stream
   │
   ├── enhance_frame()        light_correct():  gray-world WB + CLAHE on LAB's L
   │                          GFPGANer.enhance(frame, weight=strength):
   │                             face detect + restore on 512² crops
   │                             bg_upsampler = RealESRGANer (only if upscale>1)
   │                          → (enhanced_bgr, scale)
   │
   ├── [scale != 1]           alpha resized to the enhanced frame size
   │
   ├── compose_vertical()     contain / cover fit → 1080×1920 canvas,
   │                          float32 alpha blend over a flat BGR color
   │
   └── encode_video()         ffmpeg rawvideo → libx264 / yuv420p / +faststart
                              (cv2.VideoWriter mp4v fallback)
```

Sidecars written in the same pass (optional):

```
 matte/00000000.png …    8-bit alpha, SOURCE resolution
 person/00000000.png …   person color layer, same geometry
 matte/meta.json         {canvas, fit, fps, frames, background}
```

---

## 3. Technology choices

| Stage | Used | Why | Rejected |
|---|---|---|---|
| Person matting | **RVM** (Robust Video Matting, mobilenetv3, `torch.hub`) | purpose-built for video: recurrent, temporally stable, ~0.1 s/frame at 512 px | per-frame segmentation (flashes on hair/hands), chroma key (needs a greenscreen) |
| Face restore | **GFPGAN v1.4** (`arch="clean"`, `channel_multiplier=2`) | identity-consistent detail on 512² face crops; one call detects + restores + pastes back; accepts a `bg_upsampler` so SR and face restore are one pass | CodeFormer (slower, its fidelity knob duplicates `--strength`) |
| Super-resolution | **Real-ESRGAN x4plus**, `--upscale`, **off by default** | only upscaler GFPGAN integrates natively (`bg_upsampler`) | on-by-default: minutes/frame on MPS, and backends downsample faces to 96/256 anyway |
| Lighting | **classical**: gray-world WB + CLAHE on L channel, blended by `strength` | deterministic, no weights, milliseconds; fixes the two real defects (color cast, flat contrast) without inventing pixels | learned relighting: non-deterministic, heavy, out of scope |
| Composition | `_scale_fit()` + float32 alpha blend | `contain` never crops a face (safe default), `cover` fills; alpha resized with the *same* geometry so matte and person never drift | hand-rolled letterbox per caller |
| Video write | **ffmpeg** subprocess, libx264 / yuv420p / +faststart, cv2 fallback | guarantees a file `video2imgs` can ingest; OpenCV's mp4v is not reliably playable | VideoWriter only |
| GPU offload | one `EnhanceVideo` gRPC: clip in, **zip out** | the whole job is batch — per-frame RPC would be dominated by round-trips | per-frame RPC |

---

---

## 6. Measured numbers (this machine)

- MPS, default path: **~2.3 s/frame** enhance + **~0.1 s/frame** matting.
- 8 completed jobs in `data/enhance/*/job.json`, 79 frames at 1080×1920:
  **84 s / 102 s / 196 s** (≈1.1–2.5 s/frame) depending on strength.
- Background swap on saved sidecars: **1.4 s for 15 frames**.
- Disk: **479 MB for 8 jobs** (~140 MB per 79-frame job, dominated by PNG
  sidecars).

Weights: `RealESRGAN_x4plus.pth` (67 MB) + `GFPGANv1.4.pth` (348 MB) via
`scripts/download_enhance_weights.py`; face detection models auto-download on
first run from `facexlib`.


## 8. Limitations

Product:

- Flat-color background only for now
- No true relighting: lighting correction fixes exposure and color cast, it
  cannot move the key light.
---

---