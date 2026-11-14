# AGENTS.md — working rules for this repo

LiveTalking: a Flask/aiohttp real-time digital-human server (TTS → ASR features →
avatar inference → WebRTC/RTMP/virtualcam). One process, one GPU: the app runs on
the machine that does the avatar inference. A browser reaches it from elsewhere
over an SSH tunnel (`ssh -L 8010:127.0.0.1:8010`), so the app's port can stay
bound to loopback.

There is no remote worker, no gRPC transport and no deploy pipeline. Those
existed to split the app from a second CUDA box; that split is gone.

## Hard invariants

1. **Never commit secrets.** `.env`, keychain credentials and anything under
   `data/` that holds a login stay out of commits.
2. **GPU work is local and unconditional.** Avatar inference, avatar generation
   and video enhance run in this process. Do not reintroduce a mode toggle, a
   worker address, or a "where should this run" branch: there is one answer.
3. **The serving port must not be exposed to the network.** `app.py` binds
   `0.0.0.0` today; when running on a shared or firewalled host, bind loopback
   and tunnel in. Never add a firewall rule for the app port.

## Run / test

- Start the server with `./run.sh [args]` (it pins the conda interpreter and cd's
  into the repo). Never `python app.py` from the shell: the shell `python3` is a
  different venv without torch/cv2. `LT_PYTHON` overrides the interpreter path.
- Restart with `./restart.sh`; run it backgrounded, then poll
  `curl http://localhost:<port>/api/model/status`.
- Tests: `pytest tests/` with the same pinned interpreter. Some suites
  (`firefox_check.py`, `webrtc_smoke.py`, `smoke_test.py`, `fps_probe.py`,
  `media_check.py`, `admin_*_ui.py`, `kill_*_live.py`, `firefox_mdns_pref.py`)
  are live-server or live-browser checks, not unit tests — pass `--ignore=` for
  each, or pytest will block waiting on a server that is not running.
  `tests/test_audio_pacing.py` asserts a 50 Hz delivery rate and fails on a
  loaded machine; treat its failures as flaky, not regressions.
- On WSL2, `nvidia-smi` lives at `/usr/lib/wsl/lib/nvidia-smi` and is not on the
  non-interactive PATH; `LD_LIBRARY_PATH` must include `/usr/lib/wsl/lib` or
  torch reports `cuda.is_available() == False` with no other symptom.

## Map

- `app.py` — server, session creation, model lifecycle (`model_state`,
  `ensure_model_loaded`, `unload_model`, `restart_app`); `_AVATAR_MODULES` maps
  `--model` to its module.
- `registry.py` — `@register(category, name)` / `create(...)`; categories
  `stt, llm, tts, avatar, output`.
- `avatars/base_avatar.py` — `BaseAvatar` pipeline (TTS thread, ASR worker,
  inference thread, frame processor). Subclasses implement `inference_batch`,
  `paste_back_frame`, optional `get_avatar_length`. `paste_back_frame` is the
  narrowest per-frame override point and the right seam for any change to how a
  rendered frame reaches the output.
- `avatars/<name>_avatar.py` + `avatars/<name>/genavatar.py` — per-backend model
  and video→artifact boundary.
- `server/routes.py`, `server/avatar_routes.py`, `server/enhance_routes.py` —
  HTTP API; pages in `web/` (single English UI, no `-en` twins).
- `tools/enhance/` — RVM matting + Real-ESRGAN + GFPGAN base-video pipeline.
- `docs/` — `avatar-studio.md`, `api.md`, `admin_api.md`, `virtualcam_guide.md`.

## Conventions

- Avatar artifacts are model-specific and incompatible (`face_imgs/` +
  `coords.pkl` for wav2lip, `latents.pt` + `mask/` for musetalk); fail loudly on
  mismatch instead of silently degrading.
- `inference_batch` must return exactly `self.batch_size` frames; pick sources
  with `mirror_index(length, index + i)`.
- Weights live in `models/`; avatar artifacts in `data/avatars/<id>/`.
  Keep `.gitignore` patterns anchored at the repo root — an unanchored `models/`
  also excludes the vendored `avatars/musetalk/models/*.py` sources.
- Guard fp16 to cuda/mps; avoid `.half()` on CPU.
- Other agents commit this repo concurrently: run `git status` / `git log` and
  re-read files before editing; stage only paths you own, never `git add -A`.
