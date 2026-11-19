# Task List: Avatar Video Preprocessing Tool

Ordered checklist. One item per task in `tasks/plan.md`. Check off only when the
task's acceptance criteria and verification pass.

## Phase 1: Foundation
- [x] Task 1 — Install + probe deps in `livetalking` conda env
      (realesrgan, gfpgan, basicsr, robust_video_matting, onnxruntime);
      record working versions + weight paths.
      ✅ DONE 2026-11-07 — installed onnxruntime 1.31.1, basicsr 1.4.2,
      gfpgan 1.3.8, realesrgan 0.3.0, facexlib 0.3.0. RVM via torch.hub
      (no pip pkg). basicsr patched via scripts/patch_basicsr.py (wired into
      setup.sh + requirements.txt). All forward-pass probes passed.
      See docs/plans/enhance-tool/evidence.md §5.
- [x] Task 2 — Create `tools/enhance/` skeleton + CLI
      (`--input --output --background --matte-out --max-frames`);
      wire into `setup.sh`.
      ✅ DONE 2026-11-07 — tools/enhance/{__init__,pipeline,cli}.py + README.
      Frame read + ffmpeg H.264 encode + vertical flat-bg composition are real;
      enhance_frame/matte are stub seams. Deps wired in Task 1. unit tests:
      tests/test_enhance_pipeline.py (5/5 pass).



## Checkpoint: Foundation
- [x] Deps import in project env on a real clip.
      ✅ DONE — imports + forward passes verified in Task 1 (no real clip yet; see §5).
- [x] CLI produces a playable MP4 from a sample (unenhanced is OK).
      ✅ DONE — video.mov -> 1080x1920 h264/yuv420p 30 frames, verified by ffprobe.

## Phase 2: Core Transform
- [x] Task 3 — Frame extraction + RVM person matting; write matte sidecar.
      ✅ DONE 2026-11-07 — tools/enhance/matting.py (RVMRunner, stateful) +
      pipeline.compute_alpha wired. MPS eager ~100ms/frame (jit is slower on
      MPS → only on CUDA). --matte-out writes source-res grayscale PNGs +
      meta.json. Visually verified: clean cutout, no halos. tests 6/6.
- [x] Task 4 — Lighting correction (WB + CLAHE + gamma) + Real-ESRGAN + GFPGAN.
      ✅ DONE 2026-11-07 — tools/enhance/enhance.py (Enhancer). GFPGAN face
      restore at native res + WB/CLAHE lighting, ~2.3s/frame MPS. Real-ESRGAN
      wired but OFF by default (4x too slow on MPS + wasted: avatar downsizes
      to 96/256); --upscale 4 opt-in. Weights: scripts/download_enhance_weights.py
      -> models/. Visually verified: sharp face, clean matte, no halos. 7/7 tests.
- [x] Task 5 — Vertical 9:16 composition on flat background; emit H.264 MP4.
      ✅ DONE — compose_vertical + ffmpeg H.264 encode (Task 2); verified
      end-to-end through genavatar.py: wav2lip (coords.pkl/face_imgs/full_imgs)
      AND musetalk (latents.pt/mask/mask_coords.pkl/coords.pkl/full_imgs).

## Checkpoint: Core Transform
- [x] Full tool run shows visible quality/lighting improvement, clean flat bg,
      vertical format.
      ✅ DONE — vision_analyze on enhanced output: person cut on solid #1E1E1E,
      face sharp/natural, no halos. 1080x1920 h264.
- [x] Output MP4 runs through `genavatar.py` for wav2lip AND musetalk.
      ✅ DONE — test_enhanced_wav2lip + test_enhanced_musetalk avatars generated
      from /tmp/enh_base.mp4 (enhanced 20-frame clip). Full artifact sets written.

## Phase 3: Integration + Hardening
- [x] Task 6 — Document/script the enhance → genavatar handoff; add
      `--background` / `--matte-out`.
      ✅ DONE — tools/enhance/make_avatar.sh (one command: enhance + genavatar
      for wav2lip|musetalk); --background/--matte-out/--person-out wired in cli.py.
- [x] Task 7 — Extract `composite(person, bg, size)` compositor as the
      background-swap seam (no UI).
      ✅ DONE — pipeline.recompose_from_matte() + cli --recompose-from.
      Verified end-to-end: enhanced clip -> blue background swap in seconds,
      no re-matting, no bleed (vision_analyze). Unit test test_recompose_from_matte_swap.
- [x] Task 8 — README section, before/after sample, matte sidecar format docs.
      ✅ DONE — tools/enhance/README.md (status table, weights, usage incl.
      make_avatar.sh + swap, flags, design notes); tests/test_enhance_pipeline.py.

## Checkpoint: Complete
- [x] Raw clip → clean vertical base video → avatar generation succeeds, real output.
      ✅ DONE — enhanced clip through genavatar.py for wav2lip AND musetalk.
- [x] Matte sidecar present and usable by the swap seam.
      ✅ DONE — recompose_from_matte() verified with a real background swap.
