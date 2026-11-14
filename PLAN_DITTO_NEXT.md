# DITTO — NEXT SESSION TASKS (resume point: end of 2026-11-13)

## Where things stand (verified numbers, do not re-measure)

- **Box**: 10.130.104.166 port 2222 (scp needs `-P 2222`). WSL kernel
  6.18.58.1 + driver 621.24 after user's update. Env `livetalking` py3.12 at
  `~/miniconda3/envs/livetalking` with **torch 2.9.1+cu126, torchvision
  0.24.1+cu126, torchaudio 2.9.1+cu126, cuDNN 9.10.2.21 — cuDNN convs WORK
  (5/5 matrix), no SIGBUS anywhere.** The old "dxgkrnl WSL bug" was actually a
  corrupt-mismatch cuDNN 9.1.0 wheel in the torch 2.4.1 stack; the OS updates
  did not fix it, the wheel replacement did.
- **Box pip DNS is dead** (post-update resolv.conf unreadable, root-only).
  ALL installs must be downloaded on the Mac and shipped via scp. Box pip
  flags that work: `--no-index --find-links ~/wheels/...`.
- **Box LD_LIBRARY_PATH**: nvidia/*/lib dirs appended to ~/.bashrc (cublas,
  cudnn, curand, cufft, cusolver, cusparse, cuda_runtime, cuda_nvrtc). ORT
  CUDA EP now loads (providers list includes CUDAExecutionProvider).
- **App installed on box** at `~/livetalking_box` (from `wheels/` +
  `livetalking_box/`), all imports green, musetalk backend streams but only
  **~15 fps** (unet+vae 55.9 ms/frame at bs8 fp16) — user says musetalk's
  speed is irrelevant, they want Ditto.
- **App must bind 127.0.0.1 only** — app.py line ~531 patched to "127.0.0.1"
  on the box copy (NOT yet in the repo). Phase 3 will tunnel via ssh -L.
- **Server lifecycle**: user said "don't crash the box again" — the crash was
  OOM/VRAM from running a second model instance alongside the server. Rule:
  ONE model instance at a time; check `nvidia-smi` free VRAM before any probe;
  never run the sweep while the server holds a model.

## Ditto measurements (torch 2.9.1, cuDNN ON, RTX 3060) — DONE, recorded

- Full offline pipeline (512²): 15.76 s audio in 122 s, rc=0, cuDNN ON
  (`/tmp/cfg_gpu.pkl` on box = cfg with ORT models cpu, torch cuda).
- LMDM online pass (overlap_v2=70, 10 steps): **313 ms/pass = 31.3 ms/frame**
  (budget 400 ms/pass) — real-time OK.
- Per-frame chain measured END TO END with real components
  (`/tmp/probe_backend_chain.py` on box): stitch 1.6 ms, warp 180.7 ms,
  decode 84.0 ms, putback 38.9 ms (1432² frame) = **305 ms/frame ≈ 3.3 fps
  at 512²**.
- render_size=256 experiment (synthetic half-volume): warp 58.9 + decode
  26.7 = **85.5 ms/frame ≈ 11.7 fps** (2.3× cut, NOT 4× — fixed-res layers).
  User has NO 25 fps gate — "the higher the better". TRT is the next lever.

## Phase 2 code — WRITTEN, TESTED LOCALLY, SHIPPED TO BOX (imports OK)

Files in repo (all committed locally, not pushed):
- `avatars/ditto_avatar.py` — full backend: load_model (parses cfg, ORT→cpu),
  load_avatar (validates artifact), warm_up, DittoReal with upstream's exact
  streaming law (80-frame window sliding by 10, zero-seed buffer so first pass
  latches d0, _x_d_pending queue, idle frames before LMDM emits).
- `avatars/ditto/genavatar.py` — registration → artifact writer
  (f_s/, x_s_info/, full_imgs/, M_c2o.pkl, ditto.json), max_frames cap 75.
- `avatars/ditto/artifact.py` + `avatars/audio_features/pcm.py` — pre-existing
  from earlier phases, 35 tests green.
- Wiring: app.py `_warm_up_for`/`_load_backend_weights` accept ditto;
  `server/avatar_routes.py` AVATAR_GENERATORS has "ditto" (max_frames param);
  `server/task_manager.py` runs ditto genavatar; detect.py rule pre-existing
  (ditto.json first).
- Registry verified live: plugins {'avatar': ['ditto']}. Full suite:
  **102 passed, 0 failed**.
- All shipped to box `~/livetalking_box` via `/tmp/ditto_phase2.tgz` —
  imports verified there.

## NEXT TASKS (in order)

### 1. Generate a real Ditto avatar on the box (fastest proof)
- Source video: `/Users/anthonyrm/Projets/livetalking/videos/81361edc_enhanced.mp4`
  (already on box at ~/ditto-probe/real_person.mp4 — 689K, 1080×1920).
- scp the video to box, then run genavatar DIRECTLY (not through server UI):
  `cd ~/livetalking_box && python -c "from avatars.ditto.genavatar import
  generate_avatar; generate_avatar('real_person.mp4','ditto_test1',max_frames=75)"`
- Watch VRAM (nvidia-smi) — registration uses insightface+mediapipe+extractors.
- Expected: ~75 frames × (det+landmarks+motion+appearance), a few minutes.
- VERIFY: `ditto.json` exists, f_s/ has N .npy of shape (32,16,64,64),
  x_s_info/ npz, M_c2o.pkl, full_imgs/ pngs.

### 2. First wired streaming test (offline path, no WebRTC yet)
- Small driver script on box: build DittoReal parts via load_model(),
  load_avatar('ditto_test1'), then call inference_batch(0, [random feats ×
  batch_size]) and paste_back_frame, timing per batch. Compare with the
  305 ms/frame probe numbers.
- Feed REAL audio through PcmASR: read example/audio.wav as 20 ms chunks →
  put_audio_frame loop → check features flow and frames render.
- Watch for: cond handler shape mismatches (eye arrays were zeroed — check
  upstream wants real eye_open/eye_ball from registration instead; if the
  render looks dead-eyed, regenerate artifact WITH eye state and load it).

### 3. End-to-end server test with WebRTC
- Start server: `cd ~/livetalking_box && LT_PYTHON=... ./run.sh --transport
  webrtc --model ditto --avatar_id ditto_test1` (binds 127.0.0.1:8010).
- Run /tmp/phase1_gate.py (already on box, SECONDS=45 env) with AVATAR=ditto_test1.
- Report client_fps + server `actual avg final fps` lines. NOTE: infer fps
  counter is buggy upstream (millions), ignore it; use final fps + client fps.

### 4. TRT acceleration (the big lever, user explicitly wants it)

STATUS 2026-11-14 late: **decoder + warp BOTH on TRT 10, end-to-end measured.**
- Decoder: `decoder_fp16_1013.engine` (built on-box with TRT 10.13). 75.2→31.2 ms.
- Warp: rebuilt as a **plugin-free TRT-10 hybrid** — ditto's `warp_network.onnx`
  splits at its two 5-D GridSample nodes into segA/segB/segC (3 sub-engines, no
  custom plugin needed); the two bilinear grid samples run in PyTorch fp32 in
  between (`avatars/ditto/trt_warp_hybrid.py`, split by
  `scripts/ditto/warp_split_segments.py`). Why not the original plugin plan:
  the prebuilt warp engine embeds the TRT-8 `grid_sample_3d` plugin (`dlopen`s
  `libnvinfer.so.8`, which does not exist on a TRT-10 stack) AND that plugin is
  non-deterministic under GPU contention (face-melt; see
  RemiEtien/liveportrait-trt10-hybrid-warp). Same trick from that repo, adapted
  to ditto's graph (cut points identical; occlusion mask skips around gs2).
  Warp measured **124.6→24.4 ms (5.1×)**, mean|diff| vs pytorch 0.0002.
- Wired into `ditto_avatar.load_model` behind auto-detect of
  `~/ditto-weights/warp_hybrid_engines/hyb_segB.engine` (env override
  `DITTO_WARP_ENGINE_DIR`). `load_model` verified clean on-box twice.
- **Wired streaming probe (`/tmp/wired_test.py`, real audio, 512², real avatar):
  246.2 → 195.4 (decoder only) → 97.3 ms/frame (both lanes) = 4.0→5.1→10.3 fps.**
  warm_up 2.7s→0.2s. Batch path now: warp 24 + decode 31 + LMDM ~31 (remainder:
  stitch/cond/queue overhead ~10 ms).
- Server restarted on the box with both lanes (log `/tmp/ditto_server_trt10.log`,
  127.0.0.1:8010, lazy-load on connect; `load_model` verified from a shell too).
- Pitfall recorded: naive TRTWrapper-based hybrid (host round-trips + per-call
  cudaMalloc + CPU grid samples) measured SLOWER (158.8 ms) than pytorch
  (124.6 ms) — the lean pre-allocated `execute_async_v3`-on-torch-stream wrapper
  from the FLP repo is load-bearing, not an optimization. Do not regress it.
- LMDM TRT lane: **DONE.** `lmdm_v0.4_hubert.onnx` (192 MB, HF `ditto_onnx/`, all
  standard ops — no plugin) → TRT-10 engine built on-box (84 s, 194 MB, FP16 flag,
  FP32 IO) at `~/ditto-weights/lmdm_v0.4_hubert_fp16_trt10.engine`. Upstream's
  `LMDM` already speaks TRTWrapper (`setup({x, cond_frame, cond, time_cond})` →
  `pred_noise`/`x_start`), so the DDIM sampler stays in numpy and each diffusion
  step is one engine call. Measured **33.9 → 8.75 ms/step (3.9×)**, rel diff vs
  pytorch ~1e-5 (fp16 noise floor). Wired into `ditto_avatar.load_model` behind
  auto-detect of that engine (env override `DITTO_LMDM_ENGINE`).
- **RANK TRAP (cost an hour): the LMDM engine's `cond` input is 3-D `(1,80,1103)`
  but upstream's `ConditionHandler` returns 2-D `(80,1103)`.** TRT rejects the
  shape (`setInputShape nbDims` error per step) yet the run still produced
  CORRECT output — the buffer is contiguous and the byte count matches, so the
  wrapper's later `set_tensor_address` silently binds it. Fix in
  `ditto_avatar._drive_lmdm`: `if cond.ndim == 2: cond = cond[None]`. Symptom
  to recognise: hundreds of `[E] IExecutionContext::setInputShape` lines that
  do NOT break the pipeline — never ignore them as harmless noise.
- **FINAL end-to-end wired probe (real audio, real avatar, 512², all three lanes):
  246.2 → 195.4 (decoder) → 97.3 (+warp) → 73.7 ms/frame = 4.0 → 5.1 → 10.3 →
  13.6 fps.** Warmup 2.7 s → 0.3 s. Per-frame now: LMDM ~9 + warp ~24 +
  decode ~31 + stitch/cond/queue ~10. Rendered frame verified visually: coherent
  face, NO melt, NO black frame (the blue skin cast is the pre-existing RGB/BGR
  paste-back convention, unrelated to TRT).
- Server restarted on the box with all three lanes (log
  `/tmp/ditto_server_trt10_lmdm.log`, 127.0.0.1:8010). On the first connect the
  server logs all three lane lines and reaches `state: ready`.
- LMDM TRT engine is GPU+TRT-version specific: rebuild it on any new box/ TRT
  upgrade; the ONNX is the portable artifact (keep `lmdm_trt10_onnx.onnx` on the
  box at `~/ditto-weights/`).
- Downloaded on Mac, VERIFIED (3,115,257,094 B exact): scratch/trt_wheels/
  tensorrt_cu12_libs-10.13.3.9-*.whl (TRT 10.13 — WRONG GENERATION for the
  prebuilt engines, see below).
- DECISION MADE: prebuilt engines (ditto_trt_Ampere_Plus lane, TRT 8.6) beat
  building engines with 10.13. Upstream's `ditto_trt_Ampere_Plus/` = 2.08 GB,
  fp16 engines for everything except HuBERT+LMDM+motion_extractor (fp32),
  prebuilt for Ampere+ = RTX 3060 fits, NO engine building needed.
- TRT 8.6.1 python: pypi.nvidia.com has `tensorrt_libs-8.6.1-py2.py3-none-
  manylinux_2_17_x86_64.whl` (download RUNNING, session proc_5f6dad666c1b;
  ~2 GB). The `tensorrt==8.6.1` meta-pkg is an sdist that pip-installs
  tensorrt-cu12==8.6.1 FROM pypi.nvidia.com — check `pypi.nvidia.com/
  tensorrt-cu12/` for 8.6.1 cp312 wheel; else install libs + a hand-written
  tensorrt shim module.
- ALSO needed: `tensorrt-cu12==8.6.1` (bindings) from pypi.nvidia.com —
  verify presence; polygraphy NOT needed if using prebuilt engines.
- Fetch the engine lane: script `scratch/ditto/fetch_trt_lane.py` (had a
  KeyError: 'path' bug — siblings entries use different key names; fix by
  listing `s.get('path') or s.get('rfilename')` then retry). ~2.08 GB.
- On box: install TRT 8.6.1 into livetalking env (pip --no-index), scp lane
  to ~/ditto-weights/ditto_trt_Ampere_Plus/, run upstream inference.py with
  `--cfg_pkl v0.4_hubert_cfg_trt_online.pkl --data_root ~/ditto-weights` to
  verify engines load + measure warp/decode speedup. NOTE: tensorrt_utils.py
  TRTWrapper expects TRT 8 API (trt.Logger, deserialize). TRT 10 Python API
  differs — do NOT mix TRT 10 wheels with 8.6 engines.
- Then wire cfg: `_load_backend_weights('ditto')` must pick the TRT cfg when
  engines exist (env var DITTO_LANE=trt or auto-detect).
- Expected: warp+decode 305→~100 ms (fp16 engines + no kernel launch overhead),
  ~8-10 fps at 512², ~20+ fps at 256². LMDM stays pytorch (fp32 engine is 1.8GB
  build; skip).

### 5. If TRT insufficient → further levers (in order of cheapness)
- render_size=256 for real (registrar crops at 512² by default; cfg key
  crop_scale→crop box, then f_s shape (32,16,32,32)? VERIFY: f_s SHAPE IS
  FIXED (32,16,64,64) in artifact.py — a 256 render means the WARP output
  is 256², not f_s; re-check before coding).
- fp16 warp/decode via autocast (already fp16 in warp_network pytorch path).
- Batch the warp/decode across the 10-frame LMDM window (currently B=1 per
  call; upstream warps one frame at a time — batching 10 needs dynamic batch
  ONNX export, that's the engine-build path).

### 6. Cleanup before commit (repo hygiene, do at session end)
- `git status`: many modified files (AGENTS.md, PLAN_DITTO.md, README.md,
  app.py, base_avatar.py, detect.py, musetalk_avatar.py, config.py/yaml,
  docs, deleted remote/...). The remote-worker deletion (Phase 3 of the OLD
  plan) is ALREADY DONE and uncommitted. Commit strategy per skill:
  (a) commit the remote deletion + doc fixes as one commit,
  (b) commit Ditto Phase-2 files as another (avatars/ditto*, avatars/audio_
      features/pcm.py, tests/test_ditto_artifact.py, tests/test_pcm_asr.py,
      app.py wiring, server wiring, utils/__init__.py, avatars/__init__.py,
      avatars/audio_features/__init__.py, tests/test_avatar_registry.py),
  (c) commit PLAN_DITTO.md updates (Phase 0/1 results + this status).
  NEVER `git add -A`; stage explicit paths only. Run the suite before push.
- TODO code smells to fix before commit:
  - ditto_avatar.py: `warm_up` has an unused `from ... import transform_keypoint`
    inside; `render_src`/`if False` placeholder removed already in rewrite —
    double-check final file has no leftovers.
  - genavatar.py: `render_size_of_f_s = ... * 8` heuristic is wrong if f_s
    spatial dims change; compute from marker contract properly.

### 7. Known issues / traps for the next session
- `scp` WITHOUT `-P 2222` → exit 255 (port 22). Always `-P 2222`.
- Box reboot wipes /tmp (cfg pkl, scripts). Persistent copies live in
  ~/wheels/, ~/livetalking_box/, ~/ditto-probe/. The cfg builder script is
  `scratch/ditto/make_cfg_gpu.py` → box /tmp/cfg_gpu.pkl (rebuild if reboot).
- ONNX Runtime needs the LD_LIBRARY_PATH nvidia dirs (in ~/.bashrc now) or it
  silently falls back to CPU.
- mediapipe is REQUIRED for registration (landmark478/face_mesh) — it's in
  the env already.
- The app's PcmASR emits features even when silent (padded); the LMDM will
  run passes on silence — harmless but wasteful; ditto_avatar's idle path
  (held-mouth d0) covers pre-LMDM batches.
- The task_manager genavatar branch for ditto passes `max_frames` (default 75).
- musetalk fp16 runs at 17.9 fps ceiling on this GPU (measured) — the box is
  simply not a 4090; don't chase musetalk numbers.
- Server bind is 127.0.0.1 on box copy only — repo still has 0.0.0.0 (Phase 3
  decides; if tunnelling, port it into the repo).
- Box reboots by the teacher on crash ("崩了后短信给我重启吧") — expect uptime
  resets; /tmp loss; always `free -m` + `nvidia-smi` first.

## Key file paths (Mac)
- Plan: ~/Projets/livetalking/PLAN_DITTO.md (Phases §9)
- Skill: livetalking (loaded; references/ditto.md has all upstream contracts)
- Scratch harnesses: ~/.hermes/cache/scratch/ditto/ (probe_backend_chain.py,
  make_cfg_gpu.py, run_gpu_cudnn.py, phase1_gate.py, fetch_trt_lane.py,
  fetch_nvidia_wheels.py, collect_evidence.py)
- TRT wheels: ~/.hermes/cache/scratch/trt_wheels/ (10.13 libs — probably
  delete to free 3 GB), trt86_wheels/ (8.6.1 libs downloading)
- Ditto videos: ~/Projets/livetalking/videos/81361edc_enhanced.mp4
- Delivered: ~/Downloads/ditto_phase0_with_audio.mp4,
  ditto_real_person_with_audio.mp4, ditto_fullgpu_137s.mp4
