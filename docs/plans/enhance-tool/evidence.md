# Evidence Log: Avatar Video Preprocessing Tool

Raw probe output backing the feasibility table in `tasks/plan.md`. Re-run these
commands to reproduce.

## 1. Project environment

```
$ grep -n "PY=" run.sh
PY="/opt/miniconda3/envs/livetalking/bin/python"

$ /opt/miniconda3/envs/livetalking/bin/python -c "import sys; print(sys.version)"
3.12.15
```

## 2. Installed vs missing packages (project env)

```
$ /opt/miniconda3/envs/livetalking/bin/python - <<'PY'
import importlib.util as u
for m in ["torch","cv2","numpy","face_recognition","dlib","realesrgan","gfpgan","robust_video_matting","basicsr","onnxruntime","av"]:
    print(f"{m:24} {'OK' if u.find_spec(m) else 'MISSING'}")
PY
py: 3.12.15
torch                    OK
cv2                      OK
numpy                    OK
face_recognition          OK
dlib                     OK
realesrgan               MISSING
gfpgan                   MISSING
robust_video_matting     MISSING
basicsr                 MISSING
onnxruntime              MISSING
av                       OK
```

## 3. Open-source building blocks (GitHub API)

Query: `https://api.github.com/search/repositories?q=<q>&sort=stars`

- `robust video matting` → `PeterL1n/RobustVideoMatting`
  (9590 stars, GPL-3.0, PyTorch, Per-frame alpha matting).
- `face restoration video super resolution` →
  `SarvagyaGupta-19/GAN-For-Video-Enhancement` — 12-stage pipeline:
  FastDVDNet denoise → RealESRGAN super-res → GFPGAN face restore.
  Also `backblaze-b2-samples/ai-video-upscaler-restore` (Real-ESRGAN +
  GFPGAN).
- `video face preprocessing avatar` → no direct match (empty result set).

## 4. Commercial landscape (DuckDuckGo HTML)

Query: "automatically clean background lighting face video for avatar creation tool"
Top results: HeyGen, YouCam Video, Cliprise, VlogMe, Real-Time Video Filter —
all closed commercial avatar/video products. None exposes a standardized
face-only / base-video output that feeds an open lip-sync backend like
LiveTalking.

## Not yet probed (must run in Task 1)
- Actual `pip install` of realesrgan / gfpgan / basicsr / robust_video_matting /
  onnxruntime into the 3.12 env and a one-frame inference run. The plan's
  feasibility table marks these "Unverified".

## 5. Task 1 results (installed + probed 2026-11-07)

Env snapshot before changes: `tasks/evidence-pip-freeze-before.txt` (144 pkgs).
Base versions: torch 2.14.1, torchvision 0.29.1, numpy 2.5.4, python 3.12.15.

Installed:
```
onnxruntime 1.31.1   OK on import + forward
basicsr     1.4.2    OK after patch (see below)
gfpgan      1.3.8    OK
realesrgan  0.3.0    OK
facexlib    0.3.0    OK
```

`robust_video_matting` does NOT exist on PyPI (HTTP 404). The correct RVM
source is `torch.hub`:
```
torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True)
```
Weights land in `~/.cache/torch/hub/checkpoints/rvm_mobilenetv3.pth` (14.5 MB).
The PyPI `rvm` package is unrelated ("R in Python") — do not install it.

The only version conflict: **basicsr 1.4.2 imports
`torchvision.transforms.functional_tensor`**, a module removed in torchvision
0.17+. `basicsr`, `gfpgan`, `realesrgan` all failed to import until patched at
`basicsr/data/degradations.py:8`:
```
- from torchvision.transforms.functional_tensor import rgb_to_grayscale
+ from torchvision.transforms.functional import rgb_to_grayscale
```
This is automated by `scripts/patch_basicsr.py` (idempotent) and called from
`setup.sh`; `gfpgan`/`realesrgan` are added to `requirements.txt`.

Runtime probes (all passed — see `probe_enhance.py` in scratch):
- basicsr `rgb_to_grayscale` on (1,3,32,32) -> (1,1,32,32)
- Real-ESRGAN `SRVGGNetCompact` forward (1,3,32,32) -> (1,3,128,128) (4x)
- Real-ESRGAN `RealESRGANer` wrapper constructed (`enhance` present)
- GFPGAN `GFPGANv1Clean` forward -> tuple
- RVM `mobilenetv3` loaded with real weights, forward -> fgr (1,3,64,64),
  pha (1,1,64,64)

Still NOT downloaded (needed at Phase 2 runtime, not for the dep gate):
Real-ESRGAN pretrained `*,pth` and GFPGAN `GFPGANv1.4.pth` face-restore weights.
Track their hosting under Task 4 (bundle in `models/` + `scripts/` downloader).

env-diff check: `pip freeze` before/after (`tasks/evidence-pip-freeze-{before,after}.txt`)
shows only ADDED packages — nothing removed, downgraded, or changed. Core
`torch`/`cv2`/`numpy`/`face_recognition` still import and run. Wrinkle to watch:
`realesrgan` depends on `opencv-python` (5.0.0.93), so it now coexists with the
project's `opencv-python-headless` — both ship the `cv2` module; `cv2 5.0.0`
verified working (resize OK). If it ever causes lib-GUI issues on a headless
box, prefer the ONNX path or pin the headless build.

## 6. Task 3 results (RVM matting, 2026-11-07)

Device/perf benchmark at 1080p (`bench_rvm.py`):
```
device=cpu  jit=False per-frame=750ms
device=cpu  jit=True  per-frame=529ms
device=mps  jit=False per-frame=92ms    <- best
device=mps  jit=True  per-frame=162ms   <- jit is SLOWER on MPS
```
Decision: eager model everywhere; `torch.jit.script+freeze` only on CUDA.
Auto device order CUDA → MPS → CPU.

RVM API contract (from hub repo `model/model.py` + `inference.py`):
- `load`: `torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True)`
  (entrypoint name positional; `variant=` is NOT a hub kwarg — passing it raises
  `TypeError: load() missing 1 required positional argument: 'model'`).
- CALL: `fgr, pha, *rec = model(src, *rec, downsample_ratio)` with
  `src=[1,1,3,H,W]` float 0..1 RGB, `rec=[None]*4` initially.
- `pha` is returned at FULL source resolution (refiner upsamples internally).

End-to-end CLI run (video.mov, 15 frames, `--matte-out`):
```
h264 1080x1920, 15 frames, ~7.4s total (incl ~2s model load + first-frame warmup)
sidecar: 15 x grayscale PNG (1080x1620, source res) + meta.json
```
Visual verification (vision_analyze on composited frame 6 + matte 6): person
cleanly cut out onto solid #1E1E1E, no halos/bleed, correct 9:16 letterbox;
RVM matted both people in frame correctly.

Unit tests: `tests/test_enhance_pipeline.py` — 6/6 pass (adds RVM tensor
conversion tests: BGR→RGB float CHW and pha→uint8).

## 7. Task 4 results (quality enhancement, 2026-11-07)

Weights (download via `scripts/download_enhance_weights.py`):
- `models/RealESRGAN_x4plus.pth` (67 MB) — `curl` ok; `models/GFPGANv1.4.pth`
  (348 MB) — urllib dropped at 98/348 MB, recovered with `curl -fL --retry 5`.
- facexlib auto-downloaded `gfpgan/weights/detection_Resnet50_Final.pth`
  (104 MB) + `gfpgan/weights/parsing_parsenet.pth` (81 MB) on first GFPGANer use.

API contracts (read from installed sources):
- `RealESRGANer(scale, model_path, model, tile, tile_pad, pre_pad, half, device)`
  `.enhance(img, outscale) -> (out_bgr_uint8, img_mode)`.
- `GFPGANer(model_path, upscale, arch, channel_multiplier, bg_upsampler, device)`
  `.enhance(img, weight) -> (cropped_faces, restored_faces, restored_img)`.
  `bg_upsampler` = a RealESRGANer does background SR in the same call.
- GFPGANer pulls its RRDBNet from `basicsr.archs.rrdbnet_arch` (not
  `realesrgan.archs.rrdbnet_arch`, which does not exist).

Performance on MPS (macOS):
- `--upscale 4`: full 1080p frame → 4320p, minutes/frame (CLI timed out >420s
  for 3 frames). **Too slow and wasted** — avatar pipeline downsizes faces to
  96/256, so 4x is thrown away.
- `--upscale 1` (DEFAULT): GFPGAN face restore at native res, ~2.3s/frame.
- End-to-end default (5 frames, matte + enhance + compose + encode): 14.4s.

Decision: default `--upscale 1`; `--upscale 4` is an opt-in for low-res sources.

End-to-end visual check (vision_analyze on `/tmp/enhanced_frame2.png`):
person cleanly cut onto solid #1E1E1E, no halos/bleed; face sharp & natural
(no over-smoothing/plastic); both people in frame matted.

## 8. Tasks 6-8 results (integration + background swap, 2026-11-07)

Drop-in verification (Task 5 checkpoint): the enhanced 1080x1920 clip ran
through both `genavatar.py` backends with real output:
- wav2lip: `data/avatars/<id>/{coords.pkl,face_imgs/,full_imgs/}` (57s, SFD on
  MPS; self-recovered from an OOM by halving batch size)
- musetalk: `data/avatars/<id>/{latents.pt,mask/,mask_coords.pkl,coords.pkl,
  full_imgs/,avator_info.json}` (24s)

Background-swap seam (Task 7): `pipeline.recompose_from_matte()` + CLI
`--recompose-from`. The CLI now also writes a `--person-out` sidecar (BGR PNGs,
source resolution) — the matte alone can't rebuild person colors, so the swap
needs both. Verified end-to-end: enhanced clip -> solid blue background in
seconds with no re-matting; vision_analyze confirmed no bleed/halos and
preserved person detail.

Handoff (Task 6): `tools/enhance/make_avatar.sh <raw> <avatar_id>
[MODEL=wav2lip|musetalk]` — one command from raw clip to avatar artifacts.

Unit tests: 8/8 (adds `test_recompose_from_matte_swap`).

Crash note: a `--upscale 4` run (4320p) is heavy enough to destabilize a
17 GB laptop; the default path (`--upscale 1`) is not — 20 frames ran cleanly
post-crash. Keep super-res off unless the source is genuinely low-res.
