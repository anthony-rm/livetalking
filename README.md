# LiveTalking — Group Project

A real-time streaming digital human: text or audio goes in, and a lip-synced
virtual avatar answers on screen, over WebRTC, RTMP, or a virtual camera.

**This is a course project.** The repository is a school group project built on
the open-source LiveTalking engine released under the Apache License 2.0. It is
an academic exercise, not a production deployment, and it is not affiliated with
or endorsed by the upstream project. The original license text and copyright
notices are retained in [LICENSE](./LICENSE) and in the source file headers.

## What our group built

| Area | Where | What |
|------|-------|------|
| Remote GPU worker | `remote/`, `avatars/remote_avatar.py` | Run avatar inference on a separate CUDA box over gRPC while the host keeps TTS, ASR features and streaming. This machine is the host by default. See [docs/remote-gpu.md](docs/remote-gpu.md). |
| Deploy pipeline | `deploy.sh`, `scripts/deploy_remote.py` | One command ships the local working tree to the GPU box as a new release, swaps the `current` symlink, restarts the worker and verifies the revision it serves. Rollback, shared weights, no-sessions-interrupted. See [docs/remote-gpu.md](docs/remote-gpu.md#updating-the-box-deploysh). |
| Base-video preprocessing | `tools/enhance/`, `server/enhance_routes.py` | Matting (RVM), super-resolution, face restore (GFPGAN), 9:16 recomposition and background swap before avatar extraction. |
| Avatar Studio | `web/avatar.html`, `docs/avatar-studio.md` | One page that either enhances only or enhances-then-generates an avatar, sharing one job list. |
| Model lifecycle | `app.py`, `server/routes.py` | Lazy load on first connect, explicit unload and app restart, exposed as `/api/model/*` and in the admin console. |
| Apple Silicon support | `avatars/` | MPS device routing across the avatar backends and their audio feature extractors. |
| Single English UI | `web/` | The duplicated bilingual page set was collapsed into one English set. |

---

## Features

1. Multiple avatar backends: wav2lip, musetalk, ultralight (plus `remote`, which
   proxies to one of them over gRPC)
2. Voice cloning
3. Interrupting the avatar mid-sentence
4. Full-body video stitching
5. WebRTC, RTMP and virtual camera output
6. Action choreography: plays custom video when not speaking
7. Multi-concurrency
8. Custom avatars generated from an uploaded base video
9. Frontend API integration

---

## Usage Scenarios

| Scenario | Description |
|----------|-------------|
| Virtual streamer / live commerce | Uninterrupted streaming with LLM-generated scripts and choreographed actions |
| Digital human customer service | Knowledge-base-backed voice Q&A with interruption support |
| Online education / training | Recorded course material, or an API-driven instructor for live lectures |
| Voice assistant | Drive the `/human` endpoint from a speaker or app frontend |
| Exhibition / kiosk presenter | A digital presenter for venue narration |
| Batch short-video creation | Submit scripts through `/human` + `/record` instead of filming |

**Core flow**: input (text or audio) → LLM response (optional) → TTS synthesis →
real-time lip-sync inference → audio/video stream out.

---

## 1. Installation

### 1.1 macOS / Apple Silicon

```bash
git clone https://github.com/anthony-rm/livetalking.git
cd livetalking
./setup.sh          # creates the `livetalking` conda env and installs everything
```

`setup.sh` is idempotent. It installs a CPU/MPS torch build, the Python
requirements, the `face_recognition` stack via the prebuilt `dlib-bin` wheel
(compiling dlib from source takes several minutes), and runs
`scripts/patch_basicsr.py`, which is required because basicsr 1.4.2 imports a
module removed in torchvision 0.17+.

### 1.2 Linux / CUDA

```bash
git clone https://github.com/anthony-rm/livetalking.git
cd livetalking
conda create -n livetalking python=3.12
conda activate livetalking
# Match the CUDA build to your driver (check with nvidia-smi).
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# MuseTalk avatar generation needs face_recognition (dlib backend). The
# prebuilt dlib-bin wheel avoids compiling dlib from source.
pip install dlib-bin face_recognition_models "setuptools<81"
pip install --no-deps face-recognition

# basicsr 1.4.2 imports torchvision's removed functional_tensor module.
python scripts/patch_basicsr.py
```

`./run.sh` launches the server with a pinned interpreter. If the conda env is
not at `/opt/miniconda3/envs/livetalking/bin/python`, set `LT_PYTHON` (also
honoured by `setup.sh` and the scripts in `scripts/`).

---

## 2. Running the Server

### 2.1 Model weights

Weights are not in this repository. See [MODELS.md](MODELS.md) for the checkpoint
list, where each file goes, and how to fetch it.

### 2.2 Start

```bash
./run.sh --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
```

Weights load lazily on the first client connection, so startup is fast and the
server runs without a model loaded until something connects.

> The server needs TCP:8010 open, plus a UDP range for WebRTC media.
> Change the port with `--listenport`, or in `config.yaml`.

### 2.3 Web pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/index.html` | WebRTC connect, text/audio driver, recording |
| Avatar Studio | `/avatar.html` | Enhance a base video and/or generate an avatar from it |
| Admin Console | `/admin.html` | Session monitoring, global config, model controls, remote-worker setup |

### 2.4 Client access

| Method | Description |
|--------|-------------|
| Browser | Open `http://<host>:8010/index.html`, connect, then submit text |
| API | See [docs/api.md](docs/api.md) |

---

## 3. Architecture

<img src="./assets/dataflow.png" align="middle" />

**API layer** — `/human` takes text in echo (play directly) or chat (LLM) mode;
`/humanaudio` takes an audio file. Each connection gets a `sessionid`, which is
what makes concurrency work.

**Logic layer** — LLM response, TTS synthesis, and synchronous acoustic feature
extraction (mel / hubert / whisper).

**Rendering layer** — model inference produces lip-synced frames, which are
pasted back into the full video frame rather than replacing it.

**Streaming layer** — WebRTC for low latency, RTMP for restreaming, virtual
camera as a system device.

**Plugin system** — [registry.py](registry.py) registers TTS, avatar and output
modules by name, so a backend is chosen with `--model` / `--tts` rather than by
editing the pipeline.

### Two non-obvious constraints

- **RVM is recurrent.** Matting cannot be computed per-frame independently; the
  four hidden states must carry across frames, so `tools/enhance/matting.py` owns
  a stateful runner and resets state per stream.
- **Only crops cross the wire in remote mode.** `RemoteAvatar` ships the small
  predicted face crops, never full frames; paste-back and all streaming stay
  local. That is what makes the remote worker usable over a LAN.

---

## 4. API Documentation

| Document | Description |
|----------|-------------|
| [docs/api.md](docs/api.md) | WebRTC, text/audio driver, recording, action choreography |
| [docs/avatar_api.md](docs/avatar_api.md) | Avatar generation: create task, query progress, delete |
| [docs/admin_api.md](docs/admin_api.md) | Global config, session monitoring, force stop |
| [docs/avatar-studio.md](docs/avatar-studio.md) | The merged Avatar Studio page and its enhance → avatar chain |

Other project documents:

| Document | Description |
|----------|-------------|
| [MODELS.md](MODELS.md) | Checkpoint inventory and placement |
| [docs/remote-gpu.md](docs/remote-gpu.md) | Remote GPU worker over gRPC (admin UI + SSH setup) |
| [docs/remote-gpu-setup.md](docs/remote-gpu-setup.md) | Install and configure the GPU box: CUDA torch, deps, weights, SSH-only transport |
| [tools/enhance/README.md](tools/enhance/README.md) | Base-video preprocessing pipeline |

---

## 5. Docker

A [Dockerfile](./Dockerfile) is included; build it from the repository root. See
[MODELS.md](MODELS.md) for where weights need to be mounted.

---

## 6. Performance

Two numbers in the server log decide whether a session is live: `inferfps`
(model inference) and `finalfps` (frames actually sent). Both must be ≥ 25.

Stream compression costs CPU and inference costs GPU, so idle sessions scale with
CPU and speaking sessions with GPU. Raise `batch_size` only while `inferfps`
stays above the threshold — on an M2 Air in fp16 we measured `bs8=22.5 fps`,
`bs16=17.7 fps`, `bs32=14.0 fps`, which is why the default is 8.

---

## 7. Tests

```bash
./run.sh --skip-model --listenport 8099        # in one shell
python tests/smoke_test.py --port 8099          # 8 HTTP checks, no weights needed
```

| Script | Covers |
|--------|--------|
| `tests/smoke_test.py` | HTTP surface from `docs/api.md`; works with `--skip-model` |
| `tests/test_avatar_registry.py` | Avatar generator registry and parameter coercion |
| `tests/test_enhance_pipeline.py` | Enhance geometry, compositing and matte math |
| `tests/webrtc_smoke.py`, `tests/media_check.py` | Real WebRTC handshake and frame counting |
| `tests/firefox_check.py`, `tests/firefox_mdns_pref.py` | Firefox mDNS candidate resolution |

---

## 8. License

Apache License 2.0 — see [LICENSE](./LICENSE). This repository contains code
from the LiveTalking project; the original copyright headers are preserved in
every file that carries one.
