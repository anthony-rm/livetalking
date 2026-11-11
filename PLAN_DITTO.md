# Idea: Ditto as a `--model ditto` backend (quality upgrade: emotions, facial gestures, video reference)

Status: **idea / not started.** Plan only — nothing implemented, nothing run on
the GPU (per `AGENTS.md`, GPU work is user-run).

Target: add [Ditto](https://github.com/antgroup/ditto-talkinghead)
(arXiv 2411.19509, ACM MM 2025, Apache-2.0) as an avatar backend next to
`wav2lip` / `musetalk` / `ultralight`, for **emotion and expression synthesis on
a small GPU (RTX 3060 Ti, 8 GB)**.

Everything below was verified against the upstream `main` branch (cloned and
read) and the `digital-avatar/ditto-talkinghead` HuggingFace repo.

---

## 0. Verdict up front

Ditto fits this repo unusually well, better than any diffusion option looked at
so far, for four verified reasons:

1. **Upstream is ALREADY a threaded queue pipeline shaped like `BaseAvatar`.**
   `stream_pipeline_online.py` runs six worker threads over bounded queues:
   `audio2motion → motion_stitch → warp_f3d → decode_f3d → putback → writer`
   (`StreamSDK`, queue maxsize 100, `stop_event`, per-worker exception capture —
   the same pattern `avatars/base_avatar.py` implements, including the
   fail-one-thread-tear-all-down rule). This is not a research script to port;
   it is a runtime to *disassemble and merge*.
2. **Video reference is native.** `avatar_registrar` accepts an image
   **or a video**, extracts per-frame identity/motion info
   (`x_s_info_lst`, `f_s_lst`, `M_c2o_lst`), and *smooths it over the source
   frames* (`smo_k_s=13`). With a video source, `is_image_flag=False` and
   motion is applied **relative to the source video's own head motion**
   (stitch: `relative_d`, `eye_f0_mode` for video, `drive_eye` per-frame),
   cropped out, rendered, and pasted back into the ORIGINAL frame via
   `M_c2o` affine warp + mask blend — i.e. exactly the MuscTalk paste-back
   pattern (base video keeps body/scene/camera; audio drives brows, mouth,
   expression). This is the "video as reference" requirement, supported
   directly.
3. **Emotion/control is a designed feature, not a side effect.** The condition
   handler carries `emo` (an emotion class id, default 4), eye-open and eyeball
   conditioning from the source (`ch_info`), and per-frame `ctrl_info` knobs
   (delta pitch/yaw/roll, fade alpha, eye-open deltas) — map these to a
   server-side "emotion" API knob.
4. **Small models, three interchangeable runtimes.** `core/utils/load_model.py`
   dispatches on file extension: `.pth` → PyTorch, `.onnx` → onnxruntime,
   `.engine` → TensorRT. The PyTorch set is ~2 GB (lmdm 191 MB + warp 182 MB +
   decoder 222 MB + appearance/motion extractors ~0.5 GB + aux onnx landmarks +
   HuBERT ~1.5 GB); upstream ships prebuilt **`ditto_trt_Ampere_Plus` fp16
   engines** (a 3060 Ti *is* Ampere), so the fast path needs no engine build.

The one number to keep honest: **first-frame delay ≈ 2.8–3 s** (see §3), after
which the pipeline is bounded-pass streaming. Fine for a talking-avatar
server; must be stated in the UI.

---

## 1. What the model actually is (verified, with the numbers that matter)

| Item | Value (read from source) |
|---|---|
| License | Apache-2.0 code **and** checkpoint repo LICENSE. Caveat below (InsightFace) |
| Paradigm | audio → **landmark motion diffusion (LMDM)** → LivePortrait-style warp+decode → paste back |
| LMDM window | `seq_frames=80`, `overlap_v2=10` ⇒ every pass denoises an 80-frame window and emits **70 valid frames (2.8 s)**; windows overlap 10 frames with alpha blend (`fuse_alpha`) so there is no seam |
| Sampling | `sampling_timesteps=50` default (settable in `setup`), cosine schedule |
| Motion dim | 265 (=direction concat of scale/pitch/yaw/roll/t/exp/kp — LivePortrait motion space) |
| Audio encoder | **HuBERT with streaming KV-cache** (`hubert_streaming_fix_kv.onnx`, 1.46 GB); online chunking `(past=3, cur=5, fut=2)` × 40 ms = feed it 0.405 s PCM, get **5 valid feature frames** per call |
| Detector stack | InsightFace det + landmark106/203 (ONNX) + MediaPipe landmarker478 (`face_landmarker.task`) for eye state |
| Paste-back | `PutBack`: affine-warp the 512² render into the source frame with a soft mask (Cython `blend.pyx`, `pyximport` auto-build at import; a pure-numpy `PutBackNumpy` twin exists as fallback) |
| Envs tested | py3.10, `pytorch 2.5.1+cu121`, `tensorrt 8.6.1`, numpy 2.0.1, mediapipe (in conda file), A100 |
| Sizes on HF | Full repo 6.93 GB only because it ships all THREE formats; usable subsets: PyTorch ≈ 2.4 GB, ONNX ≈ 3.5 GB, TRT ≈ 3.3 GB |

Mirror-index idiom: upstream `Loader` uses the same `_mirror_index` as this
repo's `utils/image.py:mirror_index` — source frames cycle identically.

---

## 2. Environment decision — the one real fork in the plan

The repo conda env (py3.12, torch 2.14.1, numpy 2.5.4) vs Ditto pins
(py3.10, torch 2.5.1, TRT 8.6.1, mediapipe, onnxruntime):

- The Ditto models are plain small torch modules (`core/models/*.py`) — nothing
  exotic; on torch 2.14 they will very likely load. numpy 2.x is already
  upstream's own pin. **The PyTorch path can plausibly run in-process in the
  main env**, needing only: `onnxruntime(-gpu)` (aux landmarks + HuBERT) and
  `mediapipe` (landmark478) — both pure wheels, no compile.
- TensorRT 8.6.1 must NOT go into the main env (heavy, pinned, and TRT10 is
  incompatible with its plugin approach). The TRT engines are a **speed
  upgrade**, not the entry ticket.

Decision: **Phase 0 runs the vendored PyTorch path inside the main conda env on
the 3060 Ti.** If fps is short, the TRT path goes to a second-venv sidecar on
that machine (`avatars/ditto/serve.py` + a second `ssh -L` forward, same
`remote/tunnel.py` supervisor discipline as the existing worker), OR stays
in-process on the local GPU if the 3060 Ti is local. Do not build the sidecar
before the in-process number exists.

Mediapipe risk to check in Phase 0: protobuf version conflicts (the
avatar-harness project had to *disable* MediaPipe over a protobuf clash).
Mitigation: `landmark478_cfg` feeds only eye-open/eye-ball conditioning
(`use_eye_open`/`use_eye_ball` cfg flags) — if mediapipe fights the env, set
those flags off and lose blink-conditioning only, nothing structural.

---

## 3. Streaming contract — the honest latency math

From `stream_pipeline_online.py` (read line by line):

- Audio arrives from TTS as 20 ms chunks (the repo's existing path). Feed
  `wav2feat` every 5 chunks (0.1 s? no — chunksize `(3,5,2)`: slice step is
  `chunksize[1]*640` = 3200 samples = **0.2 s**, slice length 6480 =
  0.405 s) → each call yields 5 valid frames of HuBERT features, pushed into
  Ditto's `audio2motion_queue`-equivalent.
- The motion worker **waits until it has 70 fresh feature frames** before the
  first LMDM pass ⇒ **first motion frame requires 2.8 s of audio** + one 50-step
  LMDM pass (~measured in Phase 0; the model is 191 MB, the window is 80 frames —
  expect low hundreds of ms on a 3060 Ti).
- Steady state: every pass consumes 70 frames (2.8 s) of audio and emits 70
  frames (2.8 s) of motion. **If one pass < 2.8 s wall clock, the pipeline is
  sustainable real-time** with just the initial ~3 s delay. Warp/decode/stitch
  are single-frame ops (LivePortrait class ≈ 12.8 ms/frame on a 4090, expect
  ~35–45 ms on a 3060 Ti — tight but above 25 fps only if post stays lean; the
  512² decode is the cost). Downstream workers are 1 frame deep, so they add
  latency ≈ 3×(warp+decode+putback), not throughput loss, as long as each
  thread keeps its own fps ≥ 25.
- Warm-up buffer: upstream seeds the audio buffer with `overlap_v2*640` zeros;
  replicate it verbatim or the first fuse is wrong.

So: **time-to-first-frame ≈ 2.8 s + one pass ≈ 3.0–3.5 s**, then flat. This is
a *delayed* real-time, not batch — no `tools/enhance` fallback mode needed.

---

## 4. Proposed architecture (new / touched files)

### New: vendor the runtime

```
avatars/ditto/                     # vendored from upstream core/ (Apache-2.0)
  models/                          # lmdm, warp, decoder, extractors (pytorch)
  aux_models/                      # insightface_det, landmark106/203, hubert_stream, mediapipe478
  utils/                           # load_model, blend (pyx + numpy fallback), crop, get_mask
  keep upstream license header
```

The `StreamSDK` thread skeleton is NOT vendored — it is replaced by
`BaseAvatar`'s own four threads (TTS/ASR/inference/process), which already
exist. Keep: `Audio2Motion`, `MotionStitch`, `WarpF3D`, `DecodeF3D`, `PutBack`,
`AvatarRegistrar`, `ConditionHandler`, `Wav2Feat` as pure-callable classes
(they already are — each has `setup()` + `__call__`), called from the merged
plugin.

### New modules

| Path | Purpose |
|---|---|
| `avatars/ditto_avatar.py` | `@register("avatar", "ditto")` subclass of `BaseAvatar`. `inference_batch()` = drain PCM → `Wav2Feat` (every 0.2 s) → when ≥70 valid features, one LMDM pass → `stitch → warp → decode` per frame into `res_frame_queue`. `paste_back_frame` = Ditto `PutBack` (affine `M_c2o` + mask blend) OR the classic bbox paste — pick in Phase 3 and delete the loser. Silence path = `frame_list_cycle[idx]`, unchanged |
| `avatars/ditto/genavatar.py` | video → run `AvatarRegistrar` ONCE, store the artifact (§5). No VAE latents, no per-frame parsing; the extractor set runs once per frame (~fast, MPS-safe? check — Phase 0 can run genavatar against a CPU path) |
| `scripts/download_ditto_weights.py` | HF `digital-avatar/ditto-talkinghead`: fetch ONLY `ditto_pytorch/` + `ditto_cfg/v0.4_hubert_cfg_pytorch.pkl` (~2.4 GB), resumable, `models/ditto/`. TRT engines behind a `--trt` flag (~3.3 GB more) |
| `docs/ditto.md` | latency contract (§3), knobs, mediapipe/insightface caveats, measured fps table |
| `tests/test_ditto_artifact.py` | artifact write → `detect_backend` round-trip; musetalk/wav2lip unaffected |
| `tests/test_ditto_window.py` | window/fuse state machine against a fake LMDM: 70-frame emission, 10-frame fused overlap, seeding zeros, no drift over 60 s, barge-in discard |
| `tests/test_ditto_pasteback.py` | M_c2o warp+blend == `PutBackNumpy` reference |

### Touched

| Path | Change |
|---|---|
| `app.py` | `_AVATAR_MODULES['ditto']`; `_load_backend_weights` branch (load model set, no sidecar); `_warm_up_for` entry (empty first pass on silence audio) |
| `avatars/detect.py` | rule BEFORE wav2lip (shares `face_imgs`-like full frames): `ditto.json` present → `ditto`; update `KNOWN_BACKENDS` + the module docstring's layout table |
| `config.py` | `--model` help; `--ditto_emo` (0..N, default 4), `--ditto_ctrl_*` optional; nothing else — the knobs ride `AVATAR_GENERATORS` params |
| `server/avatar_routes.py` | `AVATAR_GENERATORS["ditto"]` with params: `crop_scale` (default 2.3), `crop_vy_ratio` (-0.125), `smo_k_s` (13), `sampling_timesteps` (50) |
| `requirements.txt` | `onnxruntime-gpu`, `mediapipe` — ONLY after the Phase-0 env probe; nothing else |
| `server/routes.py`, `web/admin.html` | expose ditto in model list + "emotion: N" display (config surfaces through existing endpoints) |
| `remote/worker.py` `_BACKENDS` | **defer.** Add only if/when fps on the 3060 Ti is insufficient and TRT sidecar is built; then mirror the pytorch inference math inside a `DittoBackend` like `MuseTalkBackend` |
| `docs/remote-gpu.md`, `MODELS.md` | attribution + the InsightFace caveat |

---

## 5. Avatar artifact format

`AvatarRegistrar(source_video)` output, pickled once at genavatar time
(one-time cost, seconds-to-minutes, then inference is pure):

```
data/avatars/<id>/
  ditto.json          # {version, sampling_timesteps, crop_scale, ...}
  full_imgs/          # RGB frames (paste-back + silence path) — SAME as musetalk keeps
  ditto_source.pt     # device-ported pickle:
                      #   x_s_info_lst (list, per frame: scale/pitch/yaw/roll/t/exp/kp)
                      #   f_s_lst  (appearance features per frame)
                      #   M_c2o_lst (crop-to-original affine per frame)
                      #   eye_open_lst / eye_ball_lst (mediapipe; may be empty)
```

Conventions this MUST respect (repo rules from the skill):
- **Device portability**: write every tensor through `t.detach().cpu()` (the
  `latents.pt` CUDA/MPS lesson — `tests/test_latents_device.py` pattern), and
  `torch.load(..., map_location=...)` in the loader. Hard requirement, test it.
- `get_avatar_length()` = `len(full_imgs)`; frame pick via
  `utils/image.mirror_index` (upstream's `_mirror_index` is identical, so
  index math already agrees with `process_frames`/silence behavior).
- Loud `RuntimeError` on mismatch, like the other `load_avatar`s.
- Rough size: LivePortrait appearance features are ≈ N × C×H×W floats
  (256×64×64 → ~4 MB fp32/frame) — for a 75-frame base clip ≈ 300 MB fp32,
  ~150 MB if stored fp16 (safe: features feed convs, but verify; consider
  storing only every-K-th frame and interpolating `x_s_info` upstream-style via
  `smo_k_s` — upstream itself smooths over temporally-adjacent infos).
  **Measure in Phase 3 and shrink if large**; `f_s` dominates, `M_c2o` is
  trivial.

Supports both image-driven avatars (single `full_imgs` frame, ditto invents
motion — musetalk cannot do this) and video-driven ones (per-frame x_s +
relative motion — the quality path for "emotions with a video reference").

---

## 6. Phased plan

| Phase | Work | Gate | Est. |
|---|---|---|---|
| **0 — env + hot-path probe (user-run on the 3060 Ti)** | `pip install onnxruntime-gpu mediapipe` into the conda env; clone upstream in a scratch dir; run `inference.py` (pytorch cfg, offline pkl) on the stock example; then run a 20 s wav and time: one LMDM pass, warp+decode+putback per frame, peak VRAM. Report `nvidia-smi` peak too | In-process pytorch works in the app interpreter + the fps table. FAIL case → TRT sidecar branch (adds ~3 d) | 0.5 d |
| **1 — vendor + load** | copy `core/`, strip StreamSDK threads, `load_model()`-equivalent calls; unit-load all models in a probe; numpy fallback for `blend_images_cy` active by default, pyx when it builds | `./run.sh --model ditto` boots to "weights loaded" (lazy, no avatar yet); no import break in existing backends (full pytest) | 1 d |
| **2 — artifact + detection** | `genavatar.py` (video→`ditto_source.pt`+`full_imgs`+`ditto.json`), `detect.py` rule ordered first, `AVATAR_GENERATORS` entry, downloader script | `POST /api/avatars` builds one; `GET /api/avatars` lists it; detection tests green; device-portability test green (write on CUDA path if available, else MPS) | 1 d |
| **3 — inference merge** | `ditto_avatar.py`: PCM→`Wav2Feat` (0.2 s cadence) into a feature ring buffer; ≥70-frame gate → LMDM pass → stitch/warp/decode per frame → queue; silence path; warm-up (`fix_kp_cond`, seeding zeros, `set_Nd`); `paste_back_frame` vs `PutBack` comparison | `tests/test_ditto_window.py` green on a fake LMDM; live session streams, `inferfps` logged | 2 d |
| **4 — quality + knobs** | emo id + `ctrl_info` surfaced (`--ditto_emo`, per-request param); `sampling_timesteps` 50 → try 25; asset-size shrink if `f_s` storage > ~500 MB; barge-in/interrupt discards pending pass | Side-by-side vs musetalk on same avatar (lip-sync, brows, blink); measured fps ≥ 25 sustained | 1 d |
| **5 — docs + remote (optional)** | `docs/ditto.md`; if 3060 Ti is that box AND fps short: TRT engines + sidecar per FlashHead plan; worker `_BACKENDS` entry | Admin card shows measured fps; unload frees VRAM (`nvidia-smi` user-checked) | 1 d |

**~6 days** with the pytorch path, +2–3 if the TRT sidecar becomes necessary.
Existing backends untouched throughout (registry isolation, per repo rules).

First milestone worth seeing: end of Phase 3, a 60 s session where the first
~3 s shows the silence frames, then expression+lip motion with no drift.

---

## 7. Risks / open questions

- **First-frame delay ≈ 3 s is structural** (70-frame valid window × 40 ms).
  UI must show a "connecting/warming" state; the pacing audio thread already
  exists for audio, so video silence frames cover it — but state it in
  `docs/ditto.md`, don't hide it.
- **The 3060 Ti fps is the one unknown.** LMDM is small, but warp/decode at 512²
  are the LivePortrait cost class (~40 ms/frame on this silicon — plausible,
  NOT measured); if decode is the bottleneck, the TRT fp16 decoder engine
  (Ampere-ready, no build) is the fix, OR drop render to 256² (decoder supports
  it — `crop_scale`/`max_size` are kwargs).
- **`sampling_timesteps=50` per pass**, on an 80-frame window, every 2.8 s:
  budget = one pass ≤ 2.8 s. Likely healthy margin; verify, and test 25 steps
  for headroom.
- **InsightFace license**: `det_10g.onnx`/landmark models are InsightFace
  pretrained — their terms are non-commercial for the *models*. Same school-project-OK caveat as FlashHead weights; write it in `MODELS.md`.
- **mediapipe protobuf clash** possible with the app env (`PROTO` pin issues
  surface the moment mediapipe installs). Fallback: `use_eye_open=False,
  use_eye_ball=False` (condition flags exist upstream) → lose blink
  conditioning only. Decide in Phase 0, in-repo, before vendoring.
- **One resident model** — the existing `_load_backend_weights` swap rule
  covers it; ensure the pytorch ditto set frees cleanly on `unload_model`
  (all modules held in one container object).
- **f_s storage size** — measure §5's fp16/every-K compromise; a >1 GB artifact
  per avatar is not acceptable next to musetalk's latents.
- **In-process vs worker**: inference math merged locally first; only add a
  worker `DittoBackend` (mirror `MuseTalkBackend.infer`) when a remote box
  must serve it; the artifact's `f_s` then ships on `LoadAvatar` like latents do.

## 8. Acceptance criteria

1. `--model ditto` + video-reference avatar connects, streams ≥25 fps for 60 s,
   `inferfps`/`finalfps` logged, A/V aligned end to end.
2. Time-to-first-frame measured and documented (expect ~3 s).
3. Identical WAV driven through musetalk vs ditto: brows/blink/expressions
   visibly richer on ditto; teeth/lip sharpness judged on OUR avatars, verdict
   recorded in the PR.
4. Emotion knob (`--ditto_emo`, per-avatar param) visibly changes expression.
5. `pytest tests/` green incl. 3 new tests, no GPU on the CI machine.
6. Existing backends: zero behavioral change (registry tests assert).

## References

- Upstream: https://github.com/antgroup/ditto-talkinghead (Apache-2.0),
  paper https://arxiv.org/abs/2411.19509
- Weights: https://huggingface.co/digital-avatar/ditto-talkinghead
- LivePortrait ancestor: https://github.com/KwaiVGI/LivePortrait (MIT)
  — the warp/decode lineage (cf. FasterLivePortrait ~30 fps whole-pipeline on
  Ampere with TRT)
- Related in-repo precedent for the sidecar/offline patterns: `tools/enhance/`,
  `docs/remote-gpu.md`
- Repo conventions applied: registry in `avatars/detect.py` /
  `server/avatar_routes.py:AVATAR_GENERATORS` / device-portability per
  `tests/test_latents_device.py` / `mirror_index` in `utils/image.py`
