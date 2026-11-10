# AGENTS.md — working rules for this repo

LiveTalking: a Flask/aiohttp real-time digital-human server (TTS → ASR features →
avatar inference → WebRTC/RTMP/virtualcam). Two processes may exist: the local
host (this machine, MPS/CPU) and a remote CUDA box reached over SSH/gRPC.

## Hard invariants

1. **Never connect to the remote CUDA box yourself.** Do not SSH into it, do not
   run commands on it, do not open interactive sessions. Ask the user to run any
   command and paste the output back.
2. **Never install, upgrade, or modify anything on the remote box** — no
   `pip install`, no `conda` changes, no edits to its config, env files, scripts,
   or weights. You may only *use* its compute, and only through commands the user
   chooses to execute.
3. Write code, scripts and docs locally; hand the user exact copy-paste commands
   for anything that must run on the remote host.
4. Never commit secrets. `.env`, keychain credentials and `data/remote_worker.json`
   stay out of commits.

## Run / test

- Start the server with `./run.sh [args]` (it pins the conda interpreter and cd's
  into the repo). Never `python app.py` from the shell: the shell `python3` is a
  different venv without torch/cv2. `LT_PYTHON` overrides the interpreter path.
- Restart with `./restart.sh`; run it backgrounded, then poll
  `curl http://localhost:<port>/api/model/status`.
- GPU work is off by default: `config.yaml` ships `worker_mode: local`, so
  avatar inference, avatar generation and enhance all run on this machine. The
  single switch for all three is `/admin.html` → Remote GPU Worker → mode
  (`POST /api/remote/mode`), or `--worker_mode remote` / `LT_WORKER_MODE=remote`.
  Saving a worker address alone does NOT enable it — `remote_active()` is the
  only gate. In remote mode TTS, ASR features, paste-back and streaming still
  stay local.
- Ship local changes to the box with `./deploy.sh` (`--dry-run` is strictly
  local; `--rollback`, `--avatars`, `--setup-key`, `--force`). The deploy is for
  the USER to run — an agent must not execute it against the box.
- Connection is SSH-ONLY: the worker binds `127.0.0.1` on the box and the host
  reaches it via `ssh -L` (`remote/tunnel.py`, supervised at startup). Never
  reintroduce `--host 0.0.0.0` as a default, never add a firewall rule for 50051.
  Box install steps: `docs/remote-gpu-setup.md`.
- Tests: `pytest tests/` with the same pinned interpreter. Some suites
  (`firefox_check.py`, `webrtc_smoke.py`, `smoke_test.py`) are live-server checks,
  not unit tests. `tests/test_deploy_*.py` run against a fake box and touch no
  network.

## Map

- `app.py` — server, session creation, model lifecycle (`model_state`,
  `ensure_model_loaded`, `unload_model`, `restart_app`); `_avatar_modules` dict
  maps `--model` to its module.
- `registry.py` — `@register(category, name)` / `create(...)`; categories
  `stt, llm, tts, avatar, output`.
- `avatars/base_avatar.py` — `BaseAvatar` pipeline (TTS thread, ASR worker,
  inference thread, frame processor). Subclasses implement `inference_batch`,
  `paste_back_frame`, optional `get_avatar_length`.
- `avatars/<name>_avatar.py` + `avatars/<name>/genavatar.py` — per-backend model
  and video→artifact boundary.
- `server/routes.py`, `server/avatar_routes.py`, `server/enhance_routes.py` —
  HTTP API; pages in `web/` (single English UI, no `-en` twins).
- `remote/` — gRPC worker (`remote/worker.py`, runs on the CUDA box),
  sync client (`remote/client.py`), address resolution (`remote/settings.py`).
- `tools/enhance/` — RVM matting + Real-ESRGAN + GFPGAN base-video pipeline.
- `docs/` — `remote-gpu.md` (deployment), `avatar-studio.md`, `api.md`,
  `admin_api.md`, `virtualcam_guide.md`.

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
