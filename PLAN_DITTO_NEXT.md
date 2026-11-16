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

### 3. End-to-end server test with WebRTC — DONE 2026-11-14
- Gate `/tmp/phase1_gate.py` run against the live server (`AVATAR=ditto_test1`,
  `SECONDS=45`, all three TRT-10 lanes): `/offer` → 200 in 4.8 s, 39 audio feeds,
  session streamed the full window. **Server `actual avg final fps`: 11.6–11.8
  steady (9.6 first chunk, warm-up).** The client counter printed 50.02 — that is
  the aiortc audio-track 50 Hz clock, not video fps; the millions-scale `infer
  fps` is the known upstream bug. 11.6-11.8 fps is the real streaming number
  (13.6 fps is the offline-probe ceiling).

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

### 6. Cleanup before commit — DONE 2026-11-14 ( pushed as 4 commits to origin/main)
- 272876b refactor: delete the remote GPU worker split (44 files, −9873 lines)
- d324a49 feat: add the Ditto avatar backend (17 files, +1773)
- abb9516 docs: AGENTS.md — one process, one GPU
- b3150af docs: Ditto plan (this file) + install/downloader/split scripts
- Suite green before push (102 passed, live-server/flaky files excluded);
  push verified: remote/ absent from origin/main, ditto files present.
  Original commit strategy (kept for the record):
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

# PLAN_DITTO_SESSION_6 — resume point (2026-11-16, "go for 256x256")

## OUTCOME: render_size=256 lane SHIPPED — 12.6 -> 25.0 fps delivered

The 256 lane was the last lever on the plan. It works, and it closes the
throughput gap: delivery now equals the audio content rate (50 chunks/s =
25 fps), so speech advances at real time instead of ~0.5x.

    render ceiling:  69 ms/frame (14.5 fps)  ->  27.6 ms/frame (36 fps)
    live delivered:  12.6 fps               ->  25.0 fps
    drops/100:       2-5                    ->  7-15 (encoder now the limit)
    paste-back:      14.3 ms                ->  14.0 ms

## What a 256 render actually is (the verification the old plan demanded)

`f_s` is ALWAYS stored 64-edge. The registrar's `_img_crop_to_bchw256`
resizes its extractor input to 256^2 regardless of the crop `dsize`, so
`f_s` shape does NOT follow render size. A 256 render is therefore a
RESOLUTION CHOICE, not a re-registration: bilinearly shrink the stored
volumes 64 -> 32 once at avatar load. Warp+decode are fully convolutional
(proven with a half-volume through the real weights: PSNR 23 dB vs the 512
render downsampled — same face, quarter pixels).

Render = feature_edge * 8 (the SPADE decoder upsamples 8x). NOT *16.

## The paste matrix law (proven, not assumed)

A smaller render covers the SAME crop region, so

    Mc = [ M_lin * (crop_dsize / pred_w) | b ]     # linear scales, b does not

Verified two ways: the scaled bbox equals the 512 bbox exactly (wired:
both (0,1080,0,1656)), and the pasted frame correlates 0.941 with the source
in the masked region (mean|diff| 15.7/255). A naive unscaled Mc would shrink
the face to half size in the middle of the frame.

## The bug this session's own guard caught

TRT does NOT validate `set_tensor_address` buffer sizes. Feeding 64-edge
features to a 32-edge engine does not error — the engine reads the first
quarter of the buffer and renders GARBAGE, fast (measured 20 fps of noise).
`_Engine.run` now checks shapes against `shape_of` and raises. Any future
engine/lane mismatch fails loudly instead of looking like a win.

## Engines (box ~/ditto-weights/warp_r256_engines/)

Built by scratch `export_r256.py` (B=1 warp hybrid segA/B/C + B=10 decoder,
32-edge, fp16). Verified vs pytorch BEFORE wiring: warp mean|diff| 0.0002,
decode 0.72/255, decoder batch-independence exact. segC needs STATIC shapes
stamped on the cut-adjacent value_infos — the extractor leaves dim_params
from the GridSample3D op, and TRT refuses to build segC without them.

Auto-selection: engines present -> 256 lane. `DITTO_RENDER_SIZE=512` forces
full resolution; `=256` demands the 256 engines.

## Still open (unchanged, not regressed)

- A/V residual: video advances 99.45% of real time vs audio 100% (-0.22 s per
  40 s). Present BEFORE this work, unrelated to render size. The video PTS is
  monotonic-clock derived, so the gap is dropped frames not being covered —
  the fix is to fill the gap in PTS on drop, not to touch the render.
- Encoder is now the ceiling at 720p (drops 7-15/100). `--video-scale 960` or
  1080p is now affordable on the render side.
- The sibling agent's emotion feature has a failing test
  (`test_emotion_tag_does_not_leak_past_sentence_end`): no feature row ever
  carries a tag. Proven NOT caused by the feat_queue size change (fails
  identically at maxsize 2 and 30).

# PLAN_DITTO_SESSION_5c — resume point (2026-11-16, "paste-back overlap")

## OUTCOME: the planned work was WRONG — measurement redirected it

The session opened intending to overlap paste-back with GPU work. The
LT_RENDER_TRACE (feat/outq/submit/emit timings, now in base_avatar, env
`LT_RENDER_TRACE=1`) killed that plan in one run:

    [render] feat=10651ms outq=1.7ms submit=8879ms emit=0.0ms (per 50 iters)

The inference thread spends its wait on FEATURES (10.6 s of a 12 s block);
submit is the batch compute; emit and output_queue are ~0. paste-back runs
in process_frames — its own thread — and NEVER blocks the render. Overlapping
it would have changed nothing. **Trace before optimising: the plan file's
"next lever" was written before these numbers existed.**

## What shipped instead (commit 0eceac7)

1. **feat_queue 2 -> 30 + flush purge** — the real defect the probes found.
   A burst probe (TTS-pushes-whole-sentence, output_queue undrained) showed
   maxsize=2's drop-oldest evicting 54/59 batches per sentence: the mouth
   rendered the start and teleported to the end, skipping middle WORDS while
   the audio played everything. 30 batches ≈ 12 s absorbs any sentence;
   drop-oldest remains as the monologue valve (contract test drives it past
   the bound and asserts the valve fires). flush_talk now purges feat_queue
   too — after barge-in it could hold 12 s of zombie speech.
2. **Scope verified before shipping**: the coupling check (consumer drains
   output_queue at inference rate, exactly like the real server) shows
   feat depth stays 0 with drops 0 even on a full 11.8 s burst — the
   output_queue's bounded put COUPLES ASR to the consumer, so the larger
   queue cannot build a latency backlog. It only removes the drop path.
3. Live gate: **12.6 fps delivered, 2-5 drops/100** (was 135-302 drops/100
   before the async handoff). Production-matched, sync held.

## The performance map as it now stands (all measured)

    render ceiling:  69 ms/frame -> 14.5 fps (warp 24.5 + decode 25.9 +
                     LMDM ~9 + stitch/cond/queue ~10)
    live delivered:  12.6 fps steady speech (encoder 720p H264 keeps up)
    drops:          2-5/100 (encoder sheds almost nothing now)
    A/V drift:      ~-0.15 s CONSTANT offset (paste-back+encode); not growing

Next real levers, ONLY with a trace run first:
- Whether features arrive late because the TTS is slow (edge-tts 1.6-2.1 s
  synthesis) — that wait is inherent to Send->speech, not render fps.
- The remaining -0.15 s constant offset: paste-back (~14-19 ms) + encode
  buffer; shaving it means moving the paste-back into the same thread as
  the take_staged (still NOT an overlap win, just less serial depth).
- 25 fps delivery: encode at 1080p now holds ~18.7 fps idle / burst paths;
  if the user wants full 1080p sharpness, tune --video-scale 960.

## Harnesses added (Mac ~/.hermes/cache/scratch/ditto/)

- burst_probe.py: TTS-burst behaviour; REQUIRES draining output_queue to
  mirror the real server (undrained, it reports enormous drop counts that
  the live path never produces — the coupling check proved this).
- coupling_check.py: feat depth + drop counts with inference-rate draining.

# PLAN_DITTO_SESSION_5b — resume point (2026-11-16, paste-back/queue session)

## Landed (commit f50a902, pushed)

- **Async pinned handoff** in ditto_avatar: the decode output's .cpu() was a
  SYNCHRONOUS ~273 ms/window stall (GPU idle, inference thread blocked).
  Now: uint8 conversion on GPU -> async copy into pinned memory behind a CUDA
  event -> _STAGED_FRAME sentinels through the batch contract ->
  paste_back_frame resolves them FIFO, waiting only at pixel-read time.
- **Live gate: frame drops collapsed from 135-302/100 to 1/100** — delivery
  is now producer-matched (12.4 fps at sync), the encoder no longer sheds.
  Wired probe serial: 67.4 ms/frame. Production during speech 9.2 -> ~11-12
  fps. Server on the box runs this (log /tmp/ditto_server_async.log).

## Stage attribution (fine_probe.py, median-of-3, B=10 window)

    drive_lmdm   ~90 ms | stitch/prep ~96 | warp ~55 | decode ~265
    convert+cpu  ~273 ms  (THE STALL; conversion itself is only ~20)
    = ~780 ms/window measured hot

Rejected by measurement this session (do not re-try):
- Warp threads (10 parallel hybrid chains): 0.95x — the process-wide lock +
  genuinely GPU-bound; no free lunch there.
- paste_back ThreadPool(5): 1.10-1.13x — cv2 mostly holds what it needs; a
  pool buys ~10 ms/window, not worth the complexity yet.

## NEXT LEVER (the last big one in this pipeline)

**paste-back overlap**: 14-19 ms/frame runs serially in process_frames AFTER
res_frame_queue; meanwhile the GPU sits ~35% idle waiting for the next
window's features. Move paste_back INTO the inference thread (after staging,
before queue put) or a small paste thread pool fed by the staged ring; the
blend is CPU/opencv work that can overlap the next window's warp+decode.
Expected: production -> ~14 fps (the 69 ms/frame ceiling).

## Probe harnesses (Mac ~/.hermes/cache/scratch/ditto/)

- stage_probe2.py: pipeline sub-stage timings, warp serial-vs-threaded,
  paste+resize costs.
- fine_probe.py: per-stage attribution inside inference_batch, f_s astype,
  conversion cost, paste pool scaling.

# PLAN_DITTO_SESSION_5 — resume point (2026-11-16, batched-engines session)

## Landed (commit 1555858, pushed)

- **Wired render: 73.3 -> 69.0 ms/frame** via the static-B=10 batched DECODE
  engine (`decoder_b10_fp16.engine`, from `~/ditto-weights/warp_b10_engines/`).
  config.yaml on the box runs `batch_size: 10` (matched to the LMDM window; 8
  stuttered).
- **Delivered fps during speech: 7.7 -> 18.7** — the render was no longer the
  ceiling. The bottleneck moved to the CPU VP8 encoder once audio-master
  removed the 25 fps pacing cap. `--video-codec h264 --video-scale 720`
  (box server runs with these now; log /tmp/ditto_server_720.log).
- Engine build scripts on the Mac:
  `~/.hermes/cache/scratch/ditto/export_b10.py` (B=10 ONNX export + hybrid
  split) and `build_b10.py` (standalone builder — building in the same
  process as the torch exports OOM'd the GPU).

## MEASUREMENTS THAT KILL WRONG NEXT STEPS (do not re-try)

- **B=10 warp engines: SLOWER (32.3 vs 24.5 ms/frame).** Built, verified
  numerically (mean|diff| 0.0002 vs bs=1 hybrid, batch-independence exact),
  measured, rejected. The dense-motion hourglass is compute-saturated at
  B=1 on the 3060; batching only removes launch overhead which wasn't
  there. Per-frame hybrid warp is the right lane. Engines still on the box
  (warp_b10_engines/seg*) — keep for a future GPU test, do not wire.
- **The render is now ~74 ms/frame during speech with paste-back** and
  production 9.2 fps vs delivered 18.7 — decode batching gained ~5 ms/frame
  but the per-frame warp (24.5), stitch/cond/queue (~10) and paste-back
  (14-18) dominate. To go further: batch stitch, overlap paste-back with
  warp, or cut paste-back cost (fp16/ROI-only blend is already done; the
  watermark putText at 1080x1920 costs ~2 ms — check before keeping).

## NEXT LEVERS (in value order, all un-measured)

1. **Stitch/cond batching** — stitch runs per frame on CPU (numpy, ~1 ms)
   but x_d_pending pop(0) is O(n); the queue drains 10/frame. Minor.
2. **Paste-back overlap**: the paste-back runs in process_frames AFTER the
   res_frame_queue; moving the bbox blend into inference_batch (GPU-adjacent
   thread) or threading it would hide its 14-18 ms within the 69 ms render.
3. **Encode scale tuning**: 720 is 2.25x fewer pixels than 1080; 960 (1.5x)
   would trade sharpness for a couple fps if the user prefers.
4. **fp16 paste-back** at 1080p probably unnecessary if #2 lands.

## Traps hit this session

- `torch.onnx.export` in torch 2.9 defaults `dynamo=True` which imports
  `onnxscript` (not installable on the DNS-less box) — pass `dynamo=False`.
- The LEGACY exporter refuses 5-D GridSample ("volumetric input"), so the
  warp re-export NEEDS `torch.onnx.register_custom_op_symbolic` emitting
  upstream's `GridSample3D` — the shipped ONNX carries that same custom op
  (that's why it parses with the TRT-8 plugin path upstream).
- `onnx.utils.Extractor` refuses boundary names missing from value_info;
  `shape_inference` cannot see through GridSample3D, so the two custom-op
  outputs' shapes must be added MANUALLY before extract (`make_tensor_value_info`).
  Same for segC's Sigmoid input (unknown dims) — pin it or TRT rejects the
  build ("no optimization profile").
- Build engines WITHOUT torch weights resident (separate process): the
  builder OOM'd with `resizePhysical` errors while the export script's
  182-GB-peak models were still in VRAM.
- `pkill -f "export_b10"` killed the SSH wrapper (exit 255) — always
  `pgrep` first, then kill the python pid.

# PLAN_DITTO_SESSION_4 — resume point (end of 2026-11-15)

## Landed and verified this session (pushed to origin/main)

Blue-face fix, paste-back perf, and the audio-pacing fix are all in, tested,
and running on the box (server pid on 127.0.0.1:8010, `ditto_real_v2`, state
ready). Commits: `fa19c3b` (RGB->BGR), `ccf42a3` (geometry cache),
`795303a` (pacing stale chunk).

- **Blue face: FIXED and verified numerically.** The decoder returns RGB, the
  artifact frames are BGR. At the mask core the source frame measures
  R-B=+61.7 and the pasted frame +55.9; a channel swap would invert that sign.
  musetalk is unaffected (its `decode_latents` already returns BGR at
  `avatars/musetalk/models/vae.py:107`), so the fix is ditto-only and sits in
  `paste_back_frame`. Covered by `tests/test_ditto_paste_back.py` (6 tests, no
  weights/GPU); the colour test fails without the fix, so it has teeth.
- **Paste-back perf: 41.6 -> 14.3 ms/frame** (1080x1920) by warping only the
  mapped bbox and blending with `cv2.blendLinear`. Bug caught while writing it:
  `M[:2]` is a contiguous view, so `np.ascontiguousarray` returned the SAME
  memory and the in-place bbox shift corrupted `M_c2o_list` — now `M[:2].copy()`,
  covered by a test.
- **Wired probe on HEAD code: 73.3 ms/frame** (58 batches, 580 frames, real
  audio, real avatar, all three TRT-10 lanes) — matches the recorded 73.7.

## Audio: root causes MEASURED, one fixed, one structural

Instrumented with `LT_AUDIO_TRACE=1` (logs depth / dry / real every 500 ms).

1. **FIXED — silent backlog ahead of speech.** `process_frames` enqueued
   synthetic silence (type=1) into `audio_out_queue`, building up to 200 chunks
   (4 s) of zeros that arriving TTS speech queued behind. Trace showed depth
   58->111 before the first speech chunk. Now only real audio is enqueued (the
   pacer synthesises identical zeros on demand); speech starts the tick after
   the 'start' event.
2. **FIXED — the pacer re-pushed a stale chunk.** On `queue.Empty` the code
   left `chunk` bound to the previous chunk and pushed it again, repeating the
   last 20 ms instead of inserting silence (its own docstring said otherwise).
3. **STRUCTURAL, still open — audio production is capped at 2 x render fps.**
   One batch renders `batch_size` (10) frames in ~730 ms but releases only 20
   chunks = 400 ms of speech. Measured during speech: **23.8 real chunks/s
   against the 50/s the track consumes, depth pinned at 0-1, ~40-70% of the
   track spliced silence.** Video packet delivery is fine (24.8 fps, audio
   49.7 fps at the client), so this is a CONTENT-rate mismatch: at 13.6 fps the
   picture falls ~0.46 s behind per second of speech. **No buffering change can
   fix it — the render must exceed 25 fps.**

Latency split, measured: edge-tts 1.6-2.1 s, `/human` returns in ~0 ms (TTS is
async), first frame ~2-3 s after connect (model load), and the ~2 s of silence
backlog is now gone.

## NEXT SESSION TASKS (in order)

1. **Raise render throughput past 25 fps — the only real fix for "laggy audio".**
   Levers in order: (a) `render_size=256`, which §5 flags as needing a
   registration-level recheck (`f_s` is FIXED (32,16,64,64) in artifact.py, so a
   256 render is a WARP-output change, not an `f_s` change — verify the bbox and
   `M_c2o` mapping before coding); (b) batch warp+decode across the 10-frame
   window (needs dynamic-batch engines); (c) fp16 putback. Current per-frame:
   LMDM ~9 + warp ~24 + decode ~31 + stitch/cond/queue ~10 = ~73 ms.
2. **Decide the A/V policy if 25 fps is unreachable.** Alternatives to
   stretching speech: drop video frames to keep real-time audio (audio master,
   video best-effort), which is what a video call does. Needs a user decision,
   not a code guess.
3. **Real-browser human check** of the blue face + audio (the headless gates
   prove packet rates, not what the eye sees). Chrome mic permission is still
   mandatory for ICE — never remove the `getUserMedia` call.
4. Optional: `test_audio_pacing`'s two 50 Hz count assertions are load-flaky
   (they fail on the original code too); consider relaxing them to a range or
   marking them as timing tests.

## Traps confirmed again this session

- `pkill -f "app.py --transport"` over SSH kills the SSH wrapper itself
  (exit 255, command never returns). Use
  `ss -ltnp | grep 8010` to get the pid, then `kill <pid>`.
- The box's `run.sh` hardcodes `/opt/miniconda3/...`; always pass
  `LT_PYTHON=$HOME/miniconda3/envs/livetalking/bin/python`.
- `restart.sh` is fine but leaves the foreground SSH hanging when backgrounded
  with `nohup ... &` inside a remote command — expect the local call to hit its
  timeout while the server is actually up; poll `/api/model/status` separately.
- Box uptime survived the whole session (2 d 8 h) — /tmp probes were all still
  there. Reachability is still school-LAN only.

---

# PLAN_DITTO_SESSION_3 — resume point (end of 2026-11-14, evening session)

## Current advancement (all VERIFIED, nothing below is speculation)

Ditto is a fully working streaming backend on the box (10.130.104.166:2222,
WSL2, RTX 3060):

- **Three TRT-10 lanes live** in `ditto_avatar.load_model`, auto-detected on
  the box: decoder engine (75.2→31.2 ms), plugin-free hybrid warp
  (124.6→24.4 ms; segA/B/C engines), LMDM engine (33.9→8.75 ms/step).
  End-to-end: **246.2 → 73.7 ms/frame = 4.0 → 13.6 fps** (wired probe).
- **Real browser streaming WORKS** (user-confirmed + headless-Chrome verified):
  the spinner had three stacked causes, all fixed & pushed:
  1. Chrome mDNS-obfuscates host candidates without media permission;
     box has no avahi/DNS → ICE dead. Fix: `getUserMedia({audio:true})`
     before the PeerConnection in index.html (tracks stopped immediately).
  2. aiortc never fires `failed` on a silent ICE hang → zombie session held
     the slot. Fix: 30 s ICE watchdog in `server/rtc_manager.py`.
  3. `max_session: 5` OOM-killed the box (15.7 GB RSS, ~1 GB/session host RAM).
     Fix: `max_session: 1` in config.yaml. NOTE: kill/`kill-all` endpoints
     work fine; they only "lost" to the kernel OOM kill.
- **`POST /human` fixed & verified end-to-end** (TTS-AUDIO-FLOWING): edge-tts
  died on the box's dead DNS (`speech.platform.bing.com` unresolvable; the
  endpoint itself IS reachable by IP — TLS verified with the correct Microsoft
  cert). Fix in `tts/edge.py`: `_install_dns_fallback()` pins the host to
  150.171.28.10 in-process when `getaddrinfo` fails (env override
  `EDGE_TTS_FALLBACK_IP`). Measured: 3.2 s synthesis, audio flows over WebRTC.
- **Server log shows `final fps: 25.0` with a live viewer** (render loop
  saturated; the 11.6-11.8 numbers from the audio-paced gate were not the
  render ceiling).
- Avatars on box: `ditto_real_v2` (from videos/81361edc_enhanced.mp4, 75
  frames) + `ditto_test1`. `anthony010` deleted (musetalk, backend-swap trap).
- **All of the above is pushed to origin/main** (`b5e0e08`, `f6f2c6c` + the
  earlier `272876b/d324a49/abb9516/b3150af/63c7c33`).

## LAST DISCOVERY (in flight when we stopped — NOT yet shipped/committed)

**Blue-face root cause FOUND and fix written locally (uncommitted).**
The decoder returns RGB (upstream contract); `paste_back_frame` blends it onto
`cv2.imread` BGR frames without converting → R/B swapped inside the blended
face region. Fix (in `avatars/ditto_avatar.py`, uncommitted in working tree):
convert RGB→BGR in `paste_back_frame` before the warp. The same diff also
carries a `_paste_geom_cache` optimization (bbox+matrix per frame, cut
paste-back 41.6→10.9 ms/frame at 1080×1920).
**Verify tomorrow before shipping**: confirm musetalk's path does NOT double-
convert (its `inference_batch` output convention) so the fix is ditto-only;
run the suite; ship to box; visual check.

**Audio symptoms reported by the user, diagnosis INCOMPLETE:**
- "delay after Send" — partly inherent (Ditto zero-seed ~2.8 s overlap_v2=70
  agitation + 3.2 s edge-tts synthesis), but not yet measured precisely.
- "laggy audio while talking" — suspects (from code read, not yet tested):
  the `_pace_audio` wall-clock thread (base_avatar.py:495) pushes silence when
  `audio_out_queue` empties, and the WebRTC track's `next_timestamp()` runs
  its own wall-clock in `server/webrtc.py` — two independent clocks that can
  drift/stretch. Also `audio_out_queue` is Queue(maxsize=200) ≈ 4 s, and
  inference at 13.6 fps < 25 fps pacing means the queue runs dry during speech
  → silence splices → "laggy" audio. **Next session: measure queue depth
  during speech, not guess.**

## NEXT SESSION TASKS (in order)

1. **Reconnect** — box is ONLY reachable from the school WiFi (BUAA-Mobile
   LAN); it went dark at stop because the user switched to 5G (not a reboot).
   From off-campus: SSH, the tunnel and streaming CANNOT work at all — check
   `ssh -p 2222 ... uptime` FIRST when on campus; if the box rebooted, /tmp
   logs and probes are GONE (persisted: ~/ditto-weights engines,
   ~/livetalking_box, ~/ditto-probe). /tmp/wired_test.py etc. need re-upload
   from Mac scratch. Restart the tunnel with
   `ssh -N -p 2222 -L 8010:127.0.0.1:8010 ecpkn_bac@10.130.104.166`.
2. **Ship + verify the blue-face fix** (in the uncommitted diff): confirm the
   RGB/BGR conventions per backend, run the offline wired probe, screenshot a
   frame, ship `avatars/ditto_avatar.py`, restart server, human check.
   Also flag: `tests/` has nothing covering paste-back colours — worth a small
   unit test that feeds a red render + blue frame and asserts channel order.
3. **Diagnose the laggy audio properly.** Instrument `_pace_audio`: log
   `audio_out_queue.qsize()` every 500 ms during a speech run; if it drains to
   0 during speech (likely — 13.6 fps render < 25 fps audio pacing), the fix
   is to pace the audio track from the ARRIVING chunks (burst-tolerant) or
   raise render throughput, not to add buffering. Check the two-clock drift
   (pacing thread 50 Hz vs webrtc.py next_timestamp wall clock) too.
4. **Measure the real Send→speech latency** and reduce if worthwhile:
   edge-tts 3.2 s (could synthesize while LMDM seeds), Ditto seed 2.8 s
   (structural, online cfg), queue depth. Report the split to the user.
5. **Commit + push** whatever lands (suite green first). The uncommitted
   paste-back diff must be split into two commits if both the fix and the
   geometry cache are kept: fix → "fix: ditto paste-back RGB->BGR",
   cache → "perf: cache paste-back geometry".
6. Optional (from the old plan, still true): render_size=256 (~2x more headroom),
   the 25 fps ceiling itself (putback 10.9 ms + encode), DITTO_CFG env pinning.

## Traps specific to THIS session's work (do not relearn)

- Chrome-only: the mic-permission prompt is the ICE fix — never remove the
  `getUserMedia` call; verify with a REAL browser (aiortc cannot reproduce
  mDNS/DNS issues, it sends real IPs).
- The box resolves NO DNS at all: edge-tts needs the in-process pin (already
  in mqtt... in `tts/edge.py`), and any other network call on the box needs
  the same treatment (probe with `getaddrinfo` first).
- `pgrep -f "python app.py"` matches the bash WRAPPER too; find the real PID
  via `ss -ltnp | grep 8010` (pid=) or `~/livetalking_box/livetalking.pid`.
- `max_session: 1` means ANY stale session blocks new connects ~forever if it
  doesn't ICE-complete. The 30s watchdog handles it now, but expect "Maximum
  session limit reached" right after any manual client kill + immediate retry.
- aiortc clients run on the box resolve 172.24.149.154 (WSL NAT); the Mac's
  Chrome sees 10.135.60.72 — both are correct, the connection works via the
  mic-permission fix exposing the Mac's real IP to the box.

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
