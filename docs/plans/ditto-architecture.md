# Plan: Ditto, fresh architecture on the GPU box

Status: **plan only. Nothing installed, nothing run on the box.**
Per `AGENTS.md` every box command below is for the USER to run.

Immediate deliverable (Phase 0): **Ditto running standalone on the box**,
producing an mp4 from `example/audio.wav`. Not integrated, not wired into the app.

---

## 0. Verdict

Install Ditto on the box as a **local** backend of the LiveTalking server that
runs **on the box itself**. No remote worker, no gRPC, no transport split, no
second venv.

That is now the simple option rather than the ambitious one, because the app and
Ditto resolve as **one dependency set** (§2). The whole remote layer exists to
move numbers between two machines that no longer need to be two machines, so
removing it is a net simplification, not a rewrite.

Browser access is solved by carrying the stream over SSH, not by exposing the
box. Two shapes are available (§7.4): **render on the box** and forward its own
web UI with `ssh -L 8010:127.0.0.1:8010` (default, 1.0 Mb/s measured), or
**render in the browser** from a 252-byte-per-frame motion payload (Phase 5).
The firewall forces this: it drops everything but tcp/2222 and all UDP (§3.3),
so direct WebRTC from the Mac to the box cannot work whatever we design.

Correction to the earlier version of this section: it claimed "media never
crosses the network" because there was one machine. That holds only if you sit
at the box. If you render on the box and watch from the Mac, ~1.0 Mb/s of video
*does* cross — which the measured link carries comfortably, and which §7.3 sizes
rather than assumes.

---

## 1. What changed since the previous plan

The previous version of this file assumed a Mac host driving a remote CUDA
worker over a stateful session RPC, with a second venv, a 512² JPEG reply budget
of ~1 MB/s, and a 2.8 s time-to-first-frame. Four of those premises were wrong:

| Premise | Was | Verified now |
|---|---|---|
| Box | RTX 3060 Ti, 8 GB, deps pre-installed | **RTX 3060, 12 GB**, 12 cores, 15 GB RAM, 953 GB free, **nothing installed** |
| Address | 10.130.105.203, ssh :22 | **10.130.104.166, ssh :2222** (`:22` is a *different* Windows host) |
| Latency | 2.8 s (`overlap_v2=10`, 50 steps) | **0.8 s floor** (`overlap_v2=70`, 10 steps) — §4 |
| Env | Ditto cannot share the app env | **It can** — 117 packages resolve together, §2 |
| Transport | 512² JPEG over SSH ≈ 1 MB/s | **Sized properly in §7.3:** 1.0 Mb/s real, 34 Mb/s only on synthetic noise |

The 12 GB card removes the VRAM risk that drove a whole render-size/TRT
mitigation ladder. The one-env result removes the reason for gRPC at all.

---

## 2. The finding that decides the architecture

One pip resolver run over the app's own requirements **plus** Ditto's, targeting
linux wheels:

```
numpy==1.26.4  mediapipe==0.10.21  onnxruntime-gpu==1.23.2
opencv-python-headless==4.10.0.84  scipy==1.15.0  librosa==0.10.2.post1
scikit-image==0.25.0  imageio==2.36.1  einops==0.8.1  filetype==1.2.0
Cython==3.0.11  soundfile==0.12.1  transformers<5  pyyaml  flask  aiortc
diffusers  accelerate  edge_tts  openai  dashscope  keyring  paramiko  resampy
```

→ **117 packages, no conflict, on py3.10 and py3.12 alike.**

The constraint that shapes everything: **`mediapipe` requires `numpy<2`**, so the
review env pins `numpy==1.26.4`. That is safe — the app contains no numpy-2-only
API (checked: no removed aliases, no `np.string_`), and both torch 2.5.1 and the
existing 2.14 accept 1.26.

Consequences, all of which delete work:

- **No second venv.** No sidecar, no second SSH forward, no divergent
  requirements file to keep in sync.
- **No proto change.** No `DittoStart/Push/Stop`, no `capabilities` negotiation,
  no session-state RPC.
- **No `DittoBackend` wrapper** mirroring upstream's threaded pipeline across a
  network hop. Ditto's components are called directly, in-process.
- **No frame transport.** Paste-back reads the local frame list; the render never
  leaves the process.

`mediapipe` is the only genuinely new heavy dependency and it is unavoidable:
`Source2Info.__call__` calls `_get_eye_info()` unconditionally, which imports
`mediapipe` at module scope in `core/aux_models/modules/landmark478.py`.

---

## 3. Verified facts

### 3.1 Box

| Item | Value |
|---|---|
| Host / GPU | `Carbon-Machine`, GeForce RTX 3060, **12288 MiB** |
| Driver / CUDA | 560.94, CUDA 12.6 → cu124 wheels in range |
| CPU / RAM / disk | 12 cores, 15 GB RAM, 953 GB free on `/` |
| SSH | **port 2222**, `OpenSSH_10.2p1 Ubuntu-2ubuntu3.8` |
| Conda | present (prompt is `(base)`); no `livetalking` env exists |
| Installed for this task | **nothing** |

### 3.2 Model, per lane (summed from the HF manifest, `?blobs=true`)

| Lane | Size | Notes |
|---|---|---|
| `ditto_pytorch` | **2.31 GB** | 1.46 GB of it is HuBERT onnx |
| `ditto_trt_Ampere_Plus` | **2.08 GB** | fp16 engines, Ampere-compatible — a 3060 *is* Ampere |
| `ditto_onnx` | 2.53 GB | general onnx; needs the TRT converter |

Pick **one** lane. Phase 0 uses `ditto_pytorch`.

### 3.3 Network (measured from the Mac, no login)

| Probe | Result |
|---|---|
| tcp/2222 | OPEN, 21 ms, Ubuntu sshd banner |
| tcp/22 | OPEN — **wrong machine** (`OpenSSH_for_Windows_9.5`) |
| tcp/8010, 8000, 443, 80, 50051, 3478, 1935, 5900 | timeout |
| tcp/31337 (unused control) | **timeout** ⇒ firewall is default-**drop** |
| UDP 9999, 2222 | no ICMP unreachable ⇒ UDP filtered |
| handshake latency | median 18.6 ms, **σ 24.5 ms**, max 97 ms |

The unused-port control is the load-bearing row: a port with nothing listening
also times out, so every timeout above is a firewall block, not a missing
service. `8010` stays unreachable even once the app runs.

Also confirmed: the Mac's proxy is **not** in the path to the box (no host-scoped
route to 10.130/16; traffic leaves via `en0` → `10.137.0.1`), so these are real
link numbers.

### 3.4 Configs (read out of the shipped pkl files)

| Config | `seq_frames` | `overlap_v2` | `valid_clip_len` | steps | `online_mode` |
|---|---|---|---|---|---|
| `v0.4_hubert_cfg_pytorch.pkl` | 80 | 10 | 70 | 50 | **False** |
| `v0.4_hubert_cfg_trt.pkl` | 80 | 10 | 70 | 50 | False |
| `v0.4_hubert_cfg_trt_online.pkl` | 80 | **70** | **10** | **10** | **True** |

`valid_clip_len = seq_frames - overlap_v2`. Nothing else differs materially.

---

## 4. Latency and throughput, derived from source

Written out because this is the whole risk, and because two earlier quotes in
this repo's history were wrong.

### 4.1 The window machine

`Audio2Motion.__call__` runs the LMDM over a `(1, 80, 265)` sequence, fuses the
last `fuse_length` frames of the previous result into the new one, and emits
`valid_clip_len` new frames. `StreamSDK._audio2motion_worker` will not start a
pass until `valid_clip_len` **new** feature frames have accumulated.

So **frames emitted per pass = `80 - overlap_v2`**:

- `overlap_v2=70` ⇒ 10 frames/pass ⇒ **2.5 passes/s**, 0.8 s lookahead.
- `overlap_v2=10` ⇒ 70 frames/pass ⇒ **0.36 passes/s**, 2.8 s lookahead.

Higher `overlap_v2` = **lower latency, more compute**. It is the single dial.

### 4.2 Compute cost, in units comparable across configs

Cost per second of audio = `passes_per_second × steps`:

| Config | passes/s | steps | DiT forwards/s |
|---|---|---|---|
| online (`ov=70`, 10 steps) | 2.5 | 10 | **25.0** |
| offline (`ov=10`, 50 steps) | 0.357 | 50 | **17.9** |

The online cfg needs **1.4×** the LMDM throughput in exchange for 3.5× lower
latency. Good trade, if the card can do it.

### 4.3 Time to first frame — the correction

The seed is `overlap_v2 × 640` samples of zeros → 70 feature frames. The first
pass consumes 10 and produces **no output** (it only latches
`res_kp_seq_valid_start` and `d0`). Only a full 80-frame window from the new
`local_idx` emits frames, i.e. two passes:

```
0.8 s of audio (70 seed + 10 + 10) + 2 LMDM passes + warp/decode of 10 frames
```

⇒ **TTFF ≈ 1.0–1.5 s**, of which **0.8 s is a hard floor** from the seed length.
Earlier statements of 0.4 s and of 2.8 s were both wrong; this derivation is the
one to check Phase 0 against.

### 4.4 The gate

Steady state needs one pass to finish in ≤ `valid_clip_len / 25` = **400 ms** to
hold 25 fps. Directive: **if a pass exceeds 400 ms, lower `overlap_v2`** — each
halving roughly halves the pass rate, at the cost of proportionally more latency.

### 4.5 The fixed cost

Per frame at 512²: `WarpF3D` + `DecodeF3D`. LivePortrait's cost class, independent
of LMDM tuning. 25 fps means 40 ms/frame for both together. That, not the LMDM,
is the likelier thing to bind on a 3060. **Phase 0 measures it separately.**

---

## 5. Cython: avoid it rather than build it

`core/atomic_components/putback.py` imports `..utils.blend` at **module scope**,
and `blend/__init__.py` calls `pyximport.install()` — so importing `PutBack`
triggers a runtime Cython build (needs `gcc`) or fails. Its build config also
imports `distutils`, removed in python 3.12.

Upstream ships the numpy equivalent in the same file (`PutBackNumpy`, 25 lines:
`warpAffine` the mask, `warpAffine` the render, lerp, clip, cast). Our backend
uses a local copy and **never imports upstream's `putback.py`**.

Result: no Cython at runtime, no `gcc` dependency, no `distutils`; the hot path
is 2 `warpAffine` + a lerp. Phase 0's *upstream* smoke test does need `gcc`
(it imports the Cython path) — a test dependency, not a runtime one.

---

## 6. Target architecture

```
BOX (Carbon-Machine, RTX 3060 12GB, one conda env)
────────────────────────────────────────────────────────────────
  app.py  (Flask/aiohttp, ws://0.0.0.0:8010)
    │  TTS ──► ASR features ──► avatar inference ──► streamout
    │                                │
    │                       avatars/ditto_avatar.py
    │                                │
    │            ┌───────────────────┴───────────────────┐
    │            │  Ditto components, in-process:        │
    │            │  AvatarRegistrar → ConditionHandler   │
    │            │  → Wav2Feat(hubert) → Audio2Motion    │
    │            │  → MotionStitch → WarpF3D             │
    │            │  → DecodeF3D → paste_back (numpy)     │
    │            └───────────────────────────────────────┘
    │
  browser (on the box, over the SSH tunnel)
```

Rules this shape enforces:

- **One process, one env, one GPU.** No mode toggle, no address setting, no
  capability negotiation — one place the model can run.
- **Ditto is an ordinary `avatar` plugin.** Same `@register("avatar","ditto")`
  contract as musetalk/wav2lip/ultralight, same `BaseAvatar` pipeline, same
  lazy-load-on-first-connect lifecycle. Not a special case, not a sidecar.
- **Artifacts follow the existing convention**: `data/avatars/<id>/`, detected by
  `avatars/detect.py` from directory layout, failing loudly on mismatch.
- **Model code lives in one vendored dir** (`avatars/ditto_deps/`, the upstream
  `core/` tree at a pinned commit, gitignored). `requirements.txt` gains Ditto's
  deps because it is one env now.

---

## 7. What gets deleted — and what replaces the transport

Every path in the first table exists only to bridge two machines. The second
table is the part the previous version of this plan got wrong: it said "there is
one machine, so there is no transport", which stops being true the moment you
want the box to do the rendering while the browser stays on your desk.

### 7.1 The cleanup inventory (measured, not estimated)

Counted with `wc -l` over the current tree. The repo is **29,346 lines of Python**.

| Path | Lines | Why it goes |
|---|---|---|
| `remote/` (11 modules + proto + stubs) | **2,734** | gRPC worker, tunnel, keyinstall, settings, tensors |
| `scripts/deploy_remote.py` | **1,332** | release/symlink/rsync orchestration |
| `scripts/probe_remote_pipeline.py` | **208** | probe harness for the same |
| `avatars/remote_avatar.py` | **327** | the remote plugin |
| `tools/enhance/remote_enhance.py` | **103** | enhance-over-gRPC |
| `tests/` (14 remote files + 2 fakes) | **2,618** | doubles for the above |
| `deploy.sh` | 20 | entry point to the deploy script |
| `docs/remote-gpu*.md`, remote sections of `docs/admin_api.md` | — | procedures for the deleted flow |
| `config.yaml` / `config.py` remote keys | — | `worker_mode`, `remote_worker`, `remote_backend`, `--remote_*` |
| `web/admin.html` remote card, `/api/remote/*`, `data/remote_worker.json` | — | its UI and settings |

**≈7,350 lines, 25% of the Python tree.** The remote-touching code is not a leaf:
`app.py`, `config.py`, `server/routes.py`, `server/task_manager.py`,
`server/enhance_routes.py`, `web/admin.html` and `web/avatar.html` all reference
it, so the deletion touches 7 more files that stay. That is the real cost, and
it is why this is its own commit.

Kept as *ideas*, not code: the `model_state` / `model_backend` lifecycle and the
session-kill endpoints are good and are **local** concepts — they stay.

Safety: this deletes working code, so it lands as its own commit with `git`
history intact, **after** Ditto is proven on the box (Phase 3). Not a prerequisite
for anything.

### 7.2 The one thing worth salvaging: the patch, not the pipe

The old layer had one genuinely good idea buried in it — `remote_avatar.py`
subclasses `BaseAvatar` and overrides the *narrowest* method that needs a GPU.
That is the correct seam and it survives. What was wrong was the payload: a
**512² JPEG round-trip over SSH**, which is simultaneously the most expensive
thing to send and the least information-dense.

### 7.3 Payload sizing — measured on this machine

The choice of *what* crosses the wire decides everything downstream. Measured
with `cv2.imencode` on the repo's own interpreter:

| Payload | Bytes/frame | @25 fps | Verdict |
|---|---|---|---|
| 512² RGB, **smooth face crop**, JPEG q80 | 5.1 KB | **1.0 Mb/s** | this is the honest "send pixels" number |
| 512² RGB, JPEG q90 | 6.8 KB | 1.4 Mb/s | |
| 1280² smooth (full canvas), JPEG q85 | 16.3 KB | 3.3 Mb/s | |
| 512² **noise**, JPEG q80 | 171.1 KB | 34.2 Mb/s | the *worst case* that a synthetic benchmark would report |
| **Ditto `kp`, 21×3 fp32** | **252 B** | **0.05 Mb/s** | ← the actual motion payload |
| Ditto full `x_s_info`, 133 fp32 | 532 B | 0.11 Mb/s | everything the tail needs |

Two things fall out, and they contradict what the old plan assumed:

1. **Worst-case JPEG numbers are a trap.** A random-noise benchmark says 34–140
   Mb/s and makes pixel transport look impossible; a real face crop is **1.0
   Mb/s**. Any bandwidth argument here has to be made on realistic content, or it
   argues itself into the wrong design.
2. **The motion payload is ~20,000× smaller than the pixels** — 252 bytes vs
   5.1 KB, per frame. Which means the right question is never "how do we get the
   frames across", it is "how rarely can we afford to send them".

### 7.4 Transport decision

The box is reachable **only** on tcp/2222 (§3.3: every other port times out,
including an unused control port, so it is a default-drop firewall). No WebRTC,
no raw TCP, no RTMP to the box. That constrains us to SSH-carried channels, of
which there are two:

| Option | How | Bandwidth | Latency added | Browser can be remote? |
|---|---|---|---|---|
| **A. Render in the browser** | Box sends motion only (252 B/frame), browser renders | 0.05 Mb/s | +RTT (~19 ms) | **Yes** |
| **B. Render on the box** | Box renders + MJPEG/WebRTC out, forwarded by `ssh -L` | 1.0 Mb/s | +0 (one hop) | No |
| ~~C. gRPC frames~~ (today) | 512² JPEG over a stateful session RPC | 1.0–34 Mb/s | +2 hops | No |

**Option B is the default**, because it needs no new renderer, no weight
distribution, and works today: `ssh -L 8010:127.0.0.1:8010` puts the box's own
web UI on your localhost, and its existing WebRTC/MJPEG path streams at
**1.0 Mb/s** — which the measured link carries without noticing. The firewall
problem disappears because SSH is the tunnel, and the app's port stays bound to
`127.0.0.1` on the box.

**Option A is the interesting one, and it is not a small change.** It moves warp +
decode into WebGL in the browser. That needs the appearance volume `f_s`
(8.4 MB fp32 / 4.2 MB fp16 per frame, §8.3) shipped **once per avatar at connect**
— fine, it is a download not a stream — plus a `grid_sample_3d` implementation
and the SPADE decoder in WebGL, which is a real port. Its payoff is that the box
only ever sends motion, so bandwidth stops being a variable and a browser anywhere
can drive it. **Deferred to Phase 5**, and recorded here so the seam for it is
designed in now rather than retrofitted.

The seam that makes both cheap, and which survives from the old code:

```
BaseAvatar.paste_back_frame(res_frame, idx) -> finished frame
        ├── local  (§8.4): warpAffine + mask lerp, as today
        ├── option B:      render locally, publish to the normal output
        └── option A:      return motion; a client-side renderer draws it
```

`paste_back_frame` is already the single narrowest override point
(`musetalk_avatar.py:177`, and `base_avatar.py:575` calls it inside the
inference loop), so both options are a subclass, not a restructure.


---

## 8. Integration design (Phase 2 detail)

### 8.1 Registration

`avatars/ditto_avatar.py` exposes `load_model()`, `load_avatar(id)`,
`warm_up(...)`, and `@register("avatar","ditto") class DittoAvatar(BaseAvatar)`.

We own the component wiring and do **not** call `StreamSDK`: it embeds a 6-thread
queue pipeline plus a video writer, and `BaseAvatar` already provides the
threading. Driving a thread pool from inside another thread pool is the one thing
to avoid.

### 8.2 Audio — the one real design decision

Upstream needs HuBERT features from **its own** `hubert_streaming_fix_kv.onnx`
(1.46 GB, streaming KV cache, `chunksize=(3,5,2)`: feed 6480 samples, get 5
valid 1024-dim features). This repo's `HubertASR` uses
`TencentGameMate/chinese-hubert` — a different checkpoint.

| | Shape | Verdict |
|---|---|---|
| A | `PcmASR` forwards raw 16 kHz PCM into Ditto's own `Wav2Feat` | **chosen** |
| B | Feed `HubertASR` features into the LMDM | rejected — feature-space mismatch, silent quality bugs |
| C | Reuse whisper features | rejected — the LMDM was trained on HuBERT, not whisper |

So `avatars/audio_features/pcm.py` (~30 lines): accumulate PCM, emit a 6480-sample
window per 3200-sample advance, keep the 2-chunks-per-frame contract `BaseAvatar`
expects. The `.onnx` runs on the box GPU through onnxruntime-gpu, already a dep.

**Verified against source and implemented** (`avatars/audio_features/pcm.py`,
16 tests in `tests/test_pcm_asr.py`):

| Contract | Value | Source |
|---|---|---|
| window | 6480 samples (0.405 s) | `sum(chunksize)*0.04*16000 + 80` |
| stride | 3200 samples (0.200 s) | `chunksize[1]*0.04*16000` |
| output | `[5, 1024]` | `reshape(chunksize[1], 2, 1024).mean(1)` |
| rate | **25 features/s = 1 per video frame** | 16000/3200×5 |
| valid slice | `encoding[-7:-2]` | `Wav2FeatHubert.__call__` |

Two bugs the tests caught, both worth keeping fixed:

1. **`feat_queue` is bounded (maxsize=2) and drained by the inference loop.** A
   plain `put()` deadlocks the ASR thread the moment inference falls behind —
   which starvation makes routine, since a starved batch is emitted even with no
   audio. `_emit()` now uses `put_nowait`, drops the superseded batch, and counts
   it in `dropped_batches`.
2. **`batch_size` is rarely a multiple of 5**, so features are buffered and
   sliced by row, never re-windowed per frame. Re-windowing would pair frame *n*
   with the wrong audio; slicing keeps the pairing exact.

TTFF re-derived from this contract, matching §4.3 independently: 70-feature seed
+ 2 passes × 10 features × 40 ms = **0.8 s**, i.e. the seed is the floor.

### 8.3 Artifacts

Registration is offline and per-avatar, and its output is large. `f_s` is the
appearance feature volume from `core/models/modules/appearance_feature_extractor.py`,
which reshapes to **`Bx32x16x64x64`** = 2.1 M floats = **8.4 MB/frame fp32**
(16.8 MB for a 2-frame clip; **630 MB for a 75-frame clip**). This is double the
figure the previous plan carried, because it assumed LivePortrait's 2-D
`[1,256,64,64]` rather than the 5-D volume `WarpingNetwork` actually consumes.

- Store `f_s` as **fp16** (halve it → 315 MB for 75 frames, 4.2 MB/frame).
  Every tensor through `.detach().cpu()` on write, `map_location` on read — this
  repo has been bitten twice by device-tagged artifacts
  (`tests/test_latents_device.py`).
- `x_s_info` per frame is tiny; keep all of it.
- In-process paste-back needs `full_imgs/` + per-frame `M_c2o` + the 512² mask.
- `ditto.json` is the detection key: `{version, avatar_id, render_size,
  frame_count, f_s_dtype, has_eye_state, artifact_sha256}`. Its rule goes
  **first** in `detect.py`, before the `face_imgs/` test, or a ditto avatar reads
  as wav2lip.

At 4.2 MB/frame fp16, a 75-frame avatar is ~315 MB — and clips are per-avatar, so
this is a per-avatar cost, not a per-session one. **Phase 2 decides on a
measurement**, not a guess, between: fp16 full-rate (best quality); fp16 + every
2nd frame registered with `x_s_info` interpolation (upstream already smooths with
`smo_k_s=13`, so K=2 is defensible); or a single registered frame with
`is_image_flag=True`, which makes Ditto invent head motion — more expressive than
musetalk, but it discards the source video's own head movement.

### 8.4 Paste-back

`cv2.warpAffine(render, M_c2o[idx][:2], dsize=(w,h))` + mask lerp + clip +
`uint8` — exactly `PutBackNumpy`. Silence path: `frame_list_cycle[idx]` via
`mirror_index`, same as every other backend.

---

## 9. Phases

### Phase 0 — Ditto standalone on the box (the immediate ask)

Nothing integrated. Prove the model runs; measure it.

| Step | What |
|---|---|
| 0.1 | find conda; create env `livetalking`, python 3.12 |
| 0.2 | `torch torchvision torchaudio` cu124 → 2.5.1 / 0.20.1 |
| 0.3 | pip the pinned set from §2 |
| 0.4 | clone upstream at `c3e47ee` |
| 0.5 | fetch `ditto_pytorch/**` + `ditto_cfg/**` (2.31 GB) |
| 0.6 | run `inference.py` on `example/audio.wav` + `example/image.png` |

**Gate:** an mp4 comes out, plays, lips synced. Then measure and report:

| Metric | How |
|---|---|
| RTF | wall clock ÷ output duration (`ffprobe`) |
| one LMDM pass | time `Audio2Motion.__call__` over 10 passes → median |
| warp+decode per frame | time the chain over 50 frames |
| peak VRAM | `nvidia-smi` sampled during the run |
| TTFF | timestamp of the first written frame |

**Decides:** `overlap_v2` (keep 70; lower if a pass exceeds 400 ms) and
`render_size` (512 if warp+decode ≤ 40 ms, else 256).

**Status — gate PASSED (2026-11-12), with a platform caveat.** The mp4 came out
(`tmp/result_hybrid_final.mp4.tmp.mp4`, 394 frames, 15.76 s, 25 fps, h264
1432×1432, ~5 MB; writer ran 394 frames in 2:00 with zero worker errors). Getting
there required three findings, all of which shape Phase 2:

1. **Upstream drops `device` on two loader paths.** `create_model` (torch) and
   `load_force_ori_type` (ORT) both accept `device` and never forward it to the
   module ctors, whose defaults are `device="cuda"`. Consequences on this box:
   torch modules allocated noise/timesteps on CUDA regardless of cfg, and aux
   ORT sessions requested the CUDA EP even when the cfg said `cpu`. Both patched
   in the box's `core/utils/load_model.py` (kwargs-win + inject-on-TypeError).
   Phase 2 keeps its own copy of this fix — it is a correctness bug everywhere,
   not only here (any CPU-only host hits the `.to()` mismatch crash).
2. **`torch.backends.cudnn.enabled = False` is load-bearing on this box.** Any
   cuDNN conv → SIGBUS via the flaky dxgkrnl driver (pure-torch repro, 0/3 with
   cuDNN, 4/4 without; matmul/unfold+gemm unaffected). The whole pipeline runs
   with cuDNN off at these measured speeds. This is platform mitigation, not a
   model choice — retest once Windows/WSL ships a fixed dxgkrnl.
3. **Upstream's worker threads swallow exceptions**, then the surviving
   `audio2motion` blocks forever on a no-timeout `queue.put` at
   `stream_pipeline_offline.py:413` and `close()` hangs on `join()`. The
   silent killer was pyximport failing to build `core/utils/blend/blend.pyx`
   (gcc 15 + py3.12 distutils), which killed putback and, via `stop_event`,
   every other consumer. Fix shipped: `PutBackNumpy` in place of `PutBack`
   (2 warpAffines, upstream's own numpy equivalent — exactly what the skill
   prescribed) — never import `putback.py`'s Cython path.

Measured on the box (hybrid cfg: aux=CPU, torch=CUDA, cuDNN off,
`v0.4_hubert_cfg_pytorch` offline numbers overlap_v2=10/steps=50):

| Metric | Value | Notes |
|---|---|---|
| RTF | **~7.6× slower than real time** | 120 s for 15.76 s of audio |
| LMDM pass (median of 10) | **1593-1627 ms** | 50 steps, 70 valid frames/pass |
| warp per frame (median of 50) | **131 ms** | with cuDNN off |
| decode per frame (median of 50) | **91 ms** | with cuDNN off |
| peak VRAM | **8679 MiB of 12288** | nvidia-smi during run |
| TTFF | ~30 s | registration (~28 s) + first pass |

**Phase-0 verdict on the plan's decision gates:** `overlap_v2=70` online cfg is
**not reachable real-time on this box** (a pass must fit 400 ms; the offline pass
is ~1600 ms; the online cfg's 10 steps cut that ~5× to ~320 ms/pass — marginal
against 10 frames/pass = 400 ms budget: online mode landing ≈ at the gate, to
be measured with the online kwargs in Phase 2). `render_size`: at 222 ms/frame
warp+decode, 512² misses the 40 ms/frame budget by 5.5× → **plan for
`render_size=256`** and revisit if the driver fix ever lands. Offline batch use
(produce an mp4) works at measured RTF regardless.

### Phase 1 — the app runs on the box

Install the app's own requirements into the same env, `./run.sh`, and confirm
**musetalk still works locally** — it is the incumbent and the control for "did
the dependency merge break anything". Gate: an existing avatar streams ≥25 fps.
This is what proves the §2 merge *empirically*; §2 only proves it resolves.

### Phase 2 — Ditto as a plugin

`ditto_avatar.py`, `PcmASR`, `genavatar` (registration + artifact write),
`detect.py` rule, `--model ditto`, `AVATAR_GENERATORS` entry. Gate: create a
ditto avatar from a video, connect it, stream ≥25 fps, emotion knob visibly
changes expression.

### Phase 3 — replace the transport, then delete the old one

Two steps, in this order, so the box is never unreachable:

1. **Prove option B** (§7.4): `ssh -L 8010:127.0.0.1:8010`, run the app on the
   box, watch a stream from the Mac. Gate: ≥25 fps sustained, and `ifstat`/`nettop`
   confirms ≈1 Mb/s, not tens. This needs no code change — `config.yaml` already
   sets `listenport: 8010` and `app.py:647` binds `0.0.0.0`, so the only delta is
   binding `127.0.0.1` on the box and tunnelling.
2. **Delete** §7.1's inventory in one commit: tests updated, `pytest tests/`
   green, `config.yaml` / `admin.html` cleaned, `git grep -E 'worker_mode|remote_worker|remote_avatar'` empty.

Only after Phase 2 is green, because until Ditto streams on the box there is
nothing worth tunnelling.

### Phase 4 — quality and knobs

Per-frame `ctrl_info`, barge-in reset, `sampling_timesteps` sweep, TRT lane if
the PyTorch lane is short.

### Phase 5 — motion-only transport (optional, only if a remote browser is wanted)

Ship `f_s` once per avatar at connect, send 252 B/frame, render warp+decode in
WebGL. Gate: parity with the local render at 25 fps. Explicitly **not** required
for the project to be finished — option B already works.

---

## 10. Risks

| Risk | Evidence | Mitigation |
|---|---|---|
| **Warp+decode < 40 ms at 512²** | unmeasured; LivePortrait class on a 3060 | `render_size=256` — a config value, not a rewrite |
| **One LMDM pass > 400 ms** | unmeasured | lower `overlap_v2` (costs latency) or fewer steps |
| `mediapipe` pins `numpy<2` env-wide | verified (`numpy<2`) | accepted; app is numpy-2 clean; **Phase 1 proves it runs** |
| Ditto on py3.12 | upstream tested 3.10; no 3.11+ blocker found by grep | fall back to a py3.10 env for the box (same dep set resolves) |
| `f_s` 630 MB/avatar fp32 | computed from `Bx32x16x64x64` = 2.1 M floats | fp16 → 315 MB; every-2nd-frame → 158 MB |
| Eye extractor needs mediapipe at registration | verified: called unconditionally | it is a dep now; `use_eye_open=False` only disables conditioning |
| InsightFace aux licence (non-commercial) | upstream model card | school project; recorded, not redistributed |
| Only tcp/2222 reachable | measured, incl. unused-port control | option 1: browser on the box; no app change |

---

## 11. Acceptance

1. `./run.sh --model ditto` (or an auto-detected ditto avatar) connects and
   streams ≥25 fps for 60 s, `inferfps`/`finalfps` logged.
2. TTFF measured; the §4.3 derivation replaced by the measured number.
3. `pytest tests/` green with no GPU.
4. `requirements.txt` installs into **one** env on the box, from scratch.
5. musetalk and wav2lip unchanged (registry + detection tests assert).
6. Nothing Ditto-related imports on a machine without the weights; failure is a
   loud `RuntimeError`, never a silent wrong path.
7. `git grep remote_` finds only what §7 chose to keep.
8. A stream is watchable from the Mac over `ssh -L 8010:127.0.0.1:8010` with the
   box's port bound to loopback, at ≈1 Mb/s measured on the wire (§7.4 option B).

## 12. Evaluated and rejected: FasterLivePortrait + StreamTalk

Recorded so the question is not re-litigated. Both were checked against source,
not against the READMEs.

### 12.1 FasterLivePortrait — not a face model, an implementation of Ditto's tail

Cloned `warmshao/FasterLivePortrait` (1210 stars, MIT code, pushed 2025-06-29)
and read `src/models/`. It is LivePortrait converted to ONNX/TensorRT:
`appearance_feature_extractor_model.py`, `motion_extractor_model.py`,
`warping_spade_model.py`, `stitching_model.py`, `predictor.py`. **There is no
audio path of its own** — the only audio-driven thing in it is a vendored
`JoyVASA` (and an `XPose` subtree), separate features.

So it does not replace Ditto's face stage; it re-implements the stages *after*
the motion is decided (warp + SPADE decode + paste-back), which Ditto already has.

The one useful finding: **the motion spaces are identical.** Ditto's condition
vector is LivePortrait's, verbatim — `core/atomic_components/audio2motion.py:16`
builds `scale(1), pitch(66), yaw(66), roll(66), t(3), exp(63), kp(63)`. That is
the FLP/LivePortrait implicit-keypoint format. Consequence: FLP's optimized
ONNX/TRT warp+decode **could** be swapped into Ditto's tail later as a pure
performance substitution, with no retraining and no motion-space conversion.

Why it is still not the choice now:

| Issue | Detail |
|---|---|
| Doesn't remove Ditto's LMDM | The expensive, novel part of Ditto is the latent motion diffusion model. FLP has nothing to replace it with. |
| TRT path needs a custom plugin | `predictor.py:48` `ctypes.CDLL("./checkpoints/liveportrait_onnx/libgrid_sample_3d_plugin.so")`; the `.so` must be built from `FLP/grid-sample3d-trt-plugin`, and the README pins **TensorRT 8.x** ("versions >=10.x are not compatible"). |
| CV2 CUDA build is a worse trap | The ORT route needs onnxruntime-gpu **built from source** (`grid_sample` CUDA kernel didn't land upstream) — the README documents `pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl`, i.e. a locally-built wheel. That is a multi-hour build on the box and a fork of onnxruntime to maintain. |
| Duplicate stack | ~3.1 GB of ONNX weights (421 MB `warping_spade`, 115 MB `landmark`, 113 MB `motion_extractor`, …) to reproduce what Ditto's 2.31 GB lane already ships. |
| Python pin | the prebuilt ORT wheel above is `cp310`; our env is 3.12. |

**Verdict: keep as a Phase-4 optional optimisation** for the warp/decode stage
(§10's "warp+decode < 40 ms" risk) — and only if Phase 0 measures that stage as
the bottleneck. Adopting it now would add a TensorRT-8 + custom-plugin + from-
source-ORT dependency chain to solve a problem we have not yet measured.

Note the honest version of the pitch: Ditto's own `ditto_trt_Ampere_Plus` lane
(2.08 GB, fp16 `*.engine`) already provides the TensorRT speedup with upstream's
own conversion scripts, so FLP is not even uniquely good at the thing it is good
at.

### 12.2 StreamTalk — right idea, wrong output space (SMPL-X, not pixels)

`Xiangyue-Zhang/StreamTalk` confirmed: **ECCV 2026, "Streaming Co-Speech Gesture
Generation with Key-Pose Anchoring"**, MIT, 53 stars, pushed 2026-08-28. Real and
recent, unlike the first assumption that it was a talking-head repo.

It is genuinely good at what it does: **FGD 0.383** (1-speaker) / **0.293**
(all-speaker) on BEAT2, **76 FPS on a V100 16 GB**, closed-loop streaming with
key-pose anchoring to stop long-horizon drift, part-aware DiT over hands/body/root.

But its output is **SMPL-X parameters** (`poses`, `trans`, `expressions`,
`betas`, `gender`), `.npz` — a body mesh, not a rendered frame. Wiring it to Ditto
means building the missing renderer:

```
audio ─► StreamTalk ─► SMPL-X params ─► [NOTHING HERE] ─► pixels
                                          ▲
                          needs a parametric body model renderer
                          (SMPL-X licence + mesh→image + a full-body
                           appearance source that Ditto does not have)
```

Ditto renders a **portrait**: `WarpingNetwork` deforms a 5-D feature volume
extracted from one face crop (§8.3). There is no body, no torso geometry, no
garment — nothing for `poses`/`trans` to deform. The two systems do not share an
output space, so "Ditto for the face, StreamTalk for the gestures" is not a
wiring job; it requires a third model to rasterize the body and a body-level
avatar capture Ditto has no path for.

Operational costs even if we built that:

- **SMPL-X is registration-gated** (smpl-x.is.tue.mpg.de, non-commercial
  research licence) — a manual download the script cannot fetch.
- **Speaker-specific retrieval database.** SPG retrieves a tail pose from a
  per-speaker motion DB, so the box needs BEAT2's `smplxflame_30/` +
  `train_test_split.csv` (thousands of files) — not a 2 GB weight download.
- **Its own WavLM features** (`patrickvonplaten/wavlm-libri-clean-100h-large`),
  a fourth audio feature extractor in the tree alongside Ditto's HuBERT, the
  repo's `HubertASR`, and whisper.
- One checkpoint is **`2_scott`-specific**; the all-speaker one exists but its
  retrieval DB requirement is the same.
- Env is pinned py3.9 / torch 2.0 / CUDA 11.8 — a third incompatible env.

**Verdict: rejected for this architecture.** It is a body-motion model, and
LiveTalking's avatars are upper-body/portrait talking heads driven frame-by-frame
through `inference_batch`. If full-body gesture is a real product goal, it is a
separate renderer project, not a Ditto plugin.

If the actual want behind the question is *more expressive upper-body motion from
one portrait*, the cheap version already exists inside Ditto and is scheduled:
`is_image_flag=True` (single registered frame) makes Ditto invent head/upper-body
motion instead of replaying the source video's, and per-frame `ctrl_info` (§9
Phase 4) exposes expression and pose offsets as runtime knobs.

## References

- Upstream: https://github.com/antgroup/ditto-talkinghead (Apache-2.0, pinned
  `c3e47ee`), paper https://arxiv.org/abs/2411.19509
- Weights: https://huggingface.co/digital-avatar/ditto-talkinghead
- LivePortrait lineage (warp/decode) https://github.com/KwaiVGI/LivePortrait (MIT)
- Upstream sources read for this plan: `stream_pipeline_online.py`,
  `core/atomic_components/{audio2motion,condition_handler,wav2feat,avatar_registrar,putback,cfg}.py`,
  `core/models/{lmdm,warp_network,decoder,appearance_extractor}.py`,
  `core/utils/{load_model,eye_info,blend,tensorrt_utils}.py`
