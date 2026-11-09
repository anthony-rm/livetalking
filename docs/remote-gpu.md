# Remote GPU worker (Option B) — Mac host, Windows/NVIDIA compute

Keep the LiveTalking server on the Mac (HTTP API, TTS, ASR feature extraction,
paste-back, WebRTC/RTMP/virtualcam output, recording). Move the two heavy GPU
stages to a remote Windows PC with an NVIDIA GPU:

  1. **avatar inference** — MuseTalk UNet+VAE / Wav2Lip convnet (`InferBatch`)
  2. **base-video enhance** — RVM matting + Real-ESRGAN + GFPGAN (`EnhanceVideo`)

The transport is gRPC. Tensors cross the wire as `(dtype, shape, raw bytes)` —
no pickle. See `remote/proto/inference.proto` for the full contract.

```
 Mac (host, MPS/CPU)                          Windows PC (CUDA)
 ┌──────────────────────────────────┐         ┌───────────────────────────┐
 │ app.py  (aiohttp + WebRTC)       │         │ remote.worker             │
 │ TTS ─► ASR features (whisper)    │         │  ├─ LoadModel(backend)    │
 │   │                              │  gRPC   │  ├─ LoadAvatar(id)        │
 │   └─► infer_batch ───────────────┼────────►│  ├─ InferBatch  (GPU)     │
 │         ▲ pred crops             │◄────────┼──┘                        │
 │         │                        │         │  └─ EnhanceVideo (GPU)    │
 │   paste_back (cv2, local) ─► out │         │     RVM+ESRGAN+GFPGAN     │
 └──────────────────────────────────┘         └───────────────────────────┘
```

## Setup — Windows PC (the GPU box)

**Full step-by-step: `docs/remote-gpu-pc-setup.md`.** Short version:

1. Install the same Python env (conda recommended), then CUDA-enabled torch:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

2. Clone this repo on the PC and install the app deps (`requirements.txt`), plus
   the enhance stack (`realesrgan`, `gfpgan`, and run `scripts/patch_basicsr.py`).

3. Download the model weights (same `scripts/download_*` used on the Mac), or
   copy `models/` and `data/avatars/` across. The worker only needs:
   - `models/musetalkV15/`, `models/sd-vae/` (+ `models/whisper/` if you use the
     local ASR on the worker — not required; ASR stays on the Mac)
   - `models/wav2lip.pth` (wav2lip backend)
   - `models/RealESRGAN_x4plus.pth`, `models/GFPGANv1.4.pth` (enhancer)
   - `data/avatars/<id>/` for the avatars you stream (latents/face crops)

4. Start the worker:
   ```
   python -m remote.worker --host 0.0.0.0 --port 50051
   ```
   Preload a backend with `--backend musetalk|wav2lip` (otherwise it loads lazy
   on first `LoadModel`). Open TCP 50051 in Windows Firewall for the Mac's IP.
   Health check from the Mac: `python -c "from remote.client import probe; print(probe('PC_IP:50051'))"`.

## Setup — Mac (the host)

Point the server at the worker. `--remote_worker` sets `LT_REMOTE_WORKER` for
both inference and enhancement:

```
./run.sh --model remote --remote_backend musetalk \
         --remote_worker 192.168.1.42:50051 --avatar_id my_avatar
```

or equivalently `export LT_REMOTE_WORKER=192.168.1.42:50051` and just
`--model remote --remote_backend musetalk`.

### Configuring from the admin console instead

`/admin.html` has a **Remote GPU Worker** card: enter `host:port`, press
**Test Connection** (a real gRPC `Health` call — reports the GPU name and
whether the worker is actually CUDA-capable), then **Save**.

Address resolution order:

1. address saved from the admin console → `data/remote_worker.json`
2. `--remote_worker` CLI flag / `LT_REMOTE_WORKER` env var
3. unset → remote features disabled

The saved value deliberately wins over the CLI/env default, and the card's badge
says which source is in effect ("Saved here" / "From CLI / env" / "Not
configured") so an override is never silent. **Clear** deletes the file and
falls back. The address is read when a model loads or an enhance job starts, so
a change applies to the next session/job, not to sessions already connected.

API behind the card: `GET /api/remote/status`, `POST /api/remote/save`,
`POST /api/remote/test` (all JSON, `{code,msg,data}` shape like the rest).

Enhancement is routed automatically: when an address is configured, enhance
jobs (the web "enhance" flow and `/api/enhance/task`) run on the worker; the
enhanced mp4 + matte/person sidecars are zipped back into `data/enhance/<id>/`,
so the existing status/preview/recompose/avatar routes work unchanged.
Recompose (background swap) stays local — it is cheap CPU work.

## What stays local vs remote

| Stage                              | Where | Why |
|------------------------------------|-------|-----|
| TTS                                | Mac   | low latency, streaming |
| ASR feature extraction (whisper/mel/hubert) | Mac | small model, feeds features |
| avatar inference (`inference_batch`) | PC  | the heavy UNet/VAE/convnet |
| paste_back (crop → full frame)     | Mac   | cheap cv2; keeps the small crops on the wire |
| enhance (matte + SR + face restore) | PC   | RVM/Real-ESRGAN/GFPGAN |
| enhance recompose (bg swap)        | Mac   | pure compositing, no GPU models |
| WebRTC/RTMP/virtualcam, recording  | Mac   | output devices live on the host |

## Bandwidth & latency

Per inference batch only the small pred crops cross the wire (e.g. MuseTalk
`256×256×3` uint8 per frame × batch ≈ 0.8 MB for batch 16), plus the tiny audio
features in. On a LAN this is a few ms; the worker returns crops, not full
frames, so the Mac's paste-back stays local. Keep both machines on wired
gigabit/Ethernet for real-time; run the timing gate (`inferfps`/`finalfps` ≥
25) from the server logs.

## Worker behaviour

- One backend + one avatar resident at a time (the app is single-model by
  design). `LoadAvatar` replaces the resident avatar; the `RemoteAvatar`
  class re-asserts residency on session creation, so switching back to a
  Mac-cached avatar re-ships it if the worker moved to another.
- GPU work (inference, model load, enhance) is serialised by one mutex — VRAM
  use is not reentrant.
- `Unload` (via `POST /api/model/unload` on the Mac) frees worker VRAM.
- The worker falls back to MPS/CPU if CUDA is unavailable (useful for testing
  the transport locally; it will be slow).

## Regenerating the gRPC stubs

After editing `remote/proto/inference.proto`:
```
python -m grpc_tools.protoc -I remote/proto \
    --python_out=remote/proto --grpc_python_out=remote/proto \
    remote/proto/inference.proto
```
Then re-apply the relative import (the generator emits an absolute import):
in `remote/proto/inference_pb2_grpc.py`, change
`import inference_pb2 as inference__pb2` to
`from . import inference_pb2 as inference__pb2`.

## Troubleshooting

- **`LT_REMOTE_WORKER is not set`** — pass `--remote_worker` or export the env
  var before `./run.sh`.
- **worker not reachable** — check the firewall rule, that the worker is
  listening (`Health` via `remote.client.probe`), and the IP/port.
- **`no model loaded; call LoadModel first`** — the Mac connected but the
  worker has no weights; confirm `models/` exists on the worker and the
  backend name matches `--remote_backend`.
- **slow first batch** — model load + `warm_up` happen on first session; watch
  the worker log. Later batches reuse the resident model.
