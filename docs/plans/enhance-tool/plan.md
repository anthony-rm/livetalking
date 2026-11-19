# Implementation Plan: Avatar Video Preprocessing Tool

## Overview
A standalone preprocessing tool that takes a raw talking-head upload and emits a
cleaned, standardized vertical (9:16) base video for LiveTalking: enhanced
quality (super-resolution + face restoration), corrected lighting, and a clean
flat-color background — with the person-matte kept as a sidecar so backgrounds
can be swapped later. The tool's output is a normal opaque video that
LiveTalking's existing `genavatar.py` consumes unchanged for any model.

This plan follows the confirmed intent in `docs/intent/avatar-video-preprocessing.md`.

## Architecture Decisions

- **Standalone tool, not an in-server plugin.** The tool lives outside the
  Flask/aiohttp pipeline. It reads a raw clip and writes a cleaned clip to disk.
  LiveTalking then treats that clip as an ordinary base video. This keeps the
  transform fully decoupled from the real-time server and model backends.
- **Reuse proven OSS building blocks** rather than train anything:
  - Background: Robust Video Matting (RVM), per-frame person alpha.
  - Quality: Real-ESRGAN (super-resolution) + GFPGAN (face restoration).
  - Lighting: classical auto white-balance / gray-world + CLAHE + gamma.
  - Framing / face detection: reuse `face_recognition` / dlib already in the env.
- **Person-matte is the preserved asset.** Output = opaque flat-bg vertical
  video (LiveTalking contract) + sidecar matte (person alpha frames) + metadata.
  Background swap later = composite matte over a chosen background; no re-matting.
- **Frame-based I/O.** The existing `genavatar.py` (`video2imgs`) consumes full
  PNG frame sequences; the tool works in the same frame domain and re-encodes to
  MP4 (H.264) at the end. Using frames also makes RGBA / transparent output a
  later drop-in.
- **Vertical 9:16 canvas, not a hard crop.** Scale the person to fit the short
  side of a 1080x1920 canvas and pad with the flat background, so no part of the
  person is cut. (Configurable; see Open Questions.)

## Feasibility Evidence

| Claim | Probe that produced it (command + result) | Status |
|------|------|------|
| RVM is mature and used for video background removal | GitHub API search "robust video matting" → `PeterL1n/RobustVideoMatting`, 9590 stars, GPL-3.0, PyTorch | Verified (repo exists) |
| A video pipeline already wires Real-ESRGAN + GFPGAN | GitHub API search "face restoration video super resolution" → `SarvagyaGupta-19/GAN-For-Video-Enhancement` (denoise → RealESRGAN → GFPGAN) | Verified (repo exists) |
| Project env is Python 3.12 conda `livetalking` | `run.sh` → `/opt/miniconda3/envs/livetalking/bin/python`; version 3.12.15 | Verified |
| torch, cv2, numpy, face_recognition, dlib, av already installed | importlib probe in that env | Verified |
| `realesrgan`, `gfpgan`, `robust_video_matting`, `basicsr`, `onnxruntime` are NOT installed | importlib probe in that env | Verified |
| RVM/GFPGAN/Real-ESRGAN actually run and install cleanly on macOS 3.12 | Task 1: installed onnxruntime 1.31.1, basicsr 1.4.2, gfpgan 1.3.8, realesrgan 0.3.0, facexlib 0.3.0; all forward passes passed (see `docs/plans/enhance-tool/evidence.md` §5) | **Verified** |
| `robust_video_matting` pip package does not exist | PyPI HTTP 404; RVM must load via `torch.hub`; PyPI `rvm` is unrelated | Verified |
| basicsr 1.4.2 breaks on torchvision 0.29 | Import error `functional_tensor`; fixed by `scripts/patch_basicsr.py` | Verified |
| Baseline mic: no single off-the-shelf app outputs a LiveTalking-ready standardized face clip | DuckDuckGo + GitHub searches; commercial (HeyGen/Synthesia) closed, OSS pieces separate | Verified (absence of a single tool) |

## Task List

### Phase 1: Foundation
- [ ] **Task 1: Environment + dependency probe.** Install `realesrgan`,
      `gfpgan`, `basicsr`, `robust_video_matting`, `onnxruntime` into the
      `livetalking` conda env; confirm each imports and runs one frame on sample
      video without a C-extension / version conflict. Record weights paths in
      the plan. Gate the whole feature on this passing.
- [ ] **Task 2: Project skeleton + CLI.** Create `tools/enhance/` with a CLI
      entry point accepting `--input <video> --output <video>` plus defaults for
      resolution, background color, and enhancement strength. Wire into
      `setup.sh` (idempotent) and document in README.

### Checkpoint: After Tasks 1-2
- [ ] All deps import in the project env on a real clip.
- [ ] CLI runs end-to-end on a sample (even if output is not yet enhanced) and
      writes a playable MP4.

### Phase 2: Core Transform
- [x] **Task 3: Frame extraction + person matting.** ✅ implementations in
      `tools/enhance/matting.py` (`RVMRunner`, recurrent state) + `compute_alpha`;
      `--matte-out` sidecar. MPS ~100ms/frame. Verified visually. (2026-11-07)
- [x] **Task 4: Quality enhancement.** ✅ `tools/enhance/enhance.py` (`Enhancer`).
      Lighting (WB+CLAHE) + GFPGAN face restore at native res, ~2.3s/frame MPS.
      Real-ESRGAN wired but default OFF (`--upscale 4` opt-in) — 4x from 1080p
      is wasted (avatar downsizes to 96/256) and slow on MPS. Weights via
      `scripts/download_enhance_weights.py` → models/. Verified visually.
      (2026-11-07)
- [x] **Task 5: Vertical composition + flat background.** ✅ implemented in
      Task 2 (`compose_vertical` + ffmpeg H.264); Task 5 checkpoint verified by
      feeding the enhanced MP4 through `genavatar.py` for wav2lip AND musetalk
      (both produced valid avatar artifacts). (2026-11-07)

### Checkpoint: After Tasks 3-5
- [ ] Feed a raw clip through the full tool → visible quality/lighting
      improvement and a clean flat background, vertical format.
- [ ] Output MP4 loads in `genavatar.py` for both wav2lip and musetalk without
      error (manual end-to-end one-liner).

### Phase 3: Integration + Hardening
- [x] **Task 6: LiveTalking integration.** ✅ `tools/enhance/make_avatar.sh`
      (one command: enhance → genavatar for wav2lip|musetalk);
      `--background` / `--matte-out` / `--person-out` wired. (2026-11-07)
- [x] **Task 7: Background-swap seam (future-ready).** ✅
      `pipeline.recompose_from_matte()` + `--recompose-from`; needs the matte
      AND `--person-out` color layer (matte alone can't rebuild person colors).
      Verified end-to-end with a real background swap. (2026-11-07)
- [x] **Task 8: Docs + sample run.** ✅ `tools/enhance/README.md` (status,
      weights, usage incl. make_avatar.sh + swap, flags, design notes);
      `tests/test_enhance_pipeline.py` 8/8. (2026-11-07)

### Checkpoint: Complete
- [x] Raw clip → clean vertical base video → avatar generation succeeds for at
      least one backend, verified by real output.
- [x] Matte sidecar present and usable by the swap seam.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `basicsr` / `gfpgan` / `realesrgan` pin incompatible torch or numpy on 3.12 | High — blocks everything | Task 1 installs each in isolation and probes; if basicsr conflicts, fall back to ONNX Real-ESRGAN / GFPGAN builds (no basicsr dependency) |
| RVM GPL-3.0 license contamination | Med — licensing | RVM runs as an external process/layer, not linked into the LiveTalking codebase; document the boundary. Revisit if distributing |
| Super-res is slow on CPU/MPS; long clips take minutes | Med — usability | Add `--max-frames` / `--fast` mode; document expected timing; this is a one-time op per upload |
| Enhancement over-smooths or "plastic-izes" faces | Med — quality | Keep enhancement strength configurable; default conservative; provide a `--no-face-restore` escape hatch |
| Vertical framing crops a tall person | Med — content loss | Fit-to-canvas with padding, never crop; add `--fit cover|contain` |
| Matte edge halo / jitter | Med — visual | RVM's temporal stability + optional edge erosion/dilation; keep original resolution until final resize |

## Open Questions
- Default flat background color? (proposed: neutral #1E1E1E or a light studio gray)
- Target canvas beyond 1080x1920? (4K vertical for higher-end?)
- Should the tool output the matte as RGBA PNG sequence, a grayscale matte PNG
  sequence, or a single MP4 + alpha-encoded WebM? (proposed: grayscale PNG
  sidecar, since frames are already the working format)
- Max clip length / hard cap for the automatic path?
- Weight hosting: bundle Real-ESRGAN/GFPGAN checkpoints under `models/` and add
  a downloader in `scripts/`, matching the existing musetalk weight script.
