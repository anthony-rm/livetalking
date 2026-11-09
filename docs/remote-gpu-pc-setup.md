# Configuring the Windows PC as a GPU worker

Step-by-step setup for the box that runs avatar inference + video enhancement.
Verify each numbered stage before moving on; a failure at stage N will surface
as a confusing error at stage N+2 otherwise.

Assumed: Windows 10/11 64-bit, NVIDIA GPU, a recent Game Ready or Studio driver,
and admin rights. Wav2Lip fits comfortably in 4 GB; MuseTalk's UNet+VAE needs
more headroom — 8 GB is a safe target at `batch_size 8`, and 12 GB if you want
`batch_size 16` without tuning. Measure with `nvidia-smi` during a session and
lower `--batch_size` on the Mac if you see `CUDA out of memory`.

---

## 0. Check the driver supports the CUDA build you'll install

Open PowerShell:

```powershell
nvidia-smi
```

- Note the **driver version** and **CUDA Version** in the top-right header. You
  do not need the CUDA toolkit installed — pip wheels carry their own runtime —
  but the driver must be at least as new as the wheel's CUDA version.
  For `cu124` wheels: driver ≥ 550.
- If `nvidia-smi` is not found, install/update the driver from nvidia.com first.

If the driver is older than 550, either update the driver or use `cu121` wheels
below (driver ≥ 525).

---

## 1. Install Miniconda

Download and install **Miniconda3 (Python 3.12) for Windows x86_64**:
<https://docs.conda.io/en/latest/miniconda.html>

Install "Just Me", and **keep the "Add to PATH" box unchecked** (the default).
Then open **Anaconda Prompt (miniconda3)** from the Start menu — use this shell
for every command below, never plain `cmd` or PowerShell, so `conda` resolves.

Verify:

```powershell
conda --version
```

---

## 2. Create the environment

```powershell
conda create -y -n livetalking python=3.12
conda activate livetalking
python -c "import sys; print(sys.version)"
```

Confirm it prints 3.12.x. If `conda activate` fails, run
`conda init powershell` once and reopen the shell.

---

## 3. Install CUDA-enabled PyTorch

This is the single step that makes the GPU usable. Do **not** let a plain
`pip install torch` decide — it installs the CPU build on Windows.

```powershell
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Substitute `cu121` if your driver is older than 550 (see stage 0).

**Verify before continuing** — this is the checkpoint that matters most:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: a `+cu124` version, a CUDA number (e.g. `12.4`), `True`, and your GPU
name. If `is_available()` is `False`: the driver is too old for the wheel, or you
installed the CPU build (rerun the install with the `--index-url` flag).

---

## 4. Get the repo

```powershell
cd C:\
mkdir work ; cd work
git clone <your-repo-url> livetalking
cd livetalking
```

(Or copy the folder over the network / a USB drive; `git` is only needed if you
want to pull updates later.)

Everything below assumes you are **inside the repo root** — the worker stores
relative paths (`./models`, `./data/avatars`) and, importantly, must be launched
from here (see stage 8).

---

## 5. Install the Python dependencies

```powershell
pip install -r requirements.txt
```

Then the pieces that `setup.sh` handles on the Mac but that Windows needs
explicitly:

```powershell
# edge-tts 7.2.8 is broken (403 from Bing); pin a working version.
pip install --upgrade "edge-tts>=7.3.1"

# gdown is used by the MuseTalk weight downloader.
pip install gdown
```

### Fix basicsr for the new torchvision

`basicsr 1.4.2` (pulled in by `realesrgan`) imports a module torchvision
removed in 0.17+, which breaks `realesrgan`, `gfpgan`, and `basicsr` all at
once. The repo ships an idempotent patch:

```powershell
python scripts\patch_basicsr.py
```

Expect `Patched ...\basicsr\data\degradations.py`. Re-running it is safe.

### dlib / face_recognition — **not needed on the worker**

`face_recognition`/`dlib` are only used when *generating* an avatar
(`avatars/musetalk/utils/preprocessing.py`), which happens on the Mac. The
worker only runs inference, so skip that install entirely. (If you ever want to
generate avatars on the PC too, follow the `dlib-bin` instructions commented at
the bottom of `requirements.txt`.)

### Verify the stack imports

```powershell
python -c "import torch, cv2, grpc, diffusers, transformers, realesrgan, gfpgan, basicsr; print('deps OK')"
python -m grpc_tools.protoc --version
```

If `realesrgan`/`gfpgan` raise `ModuleNotFoundError: torchvision.transforms.functional_tensor`,
the basicsr patch did not apply — rerun stage 5's patch step.

---

## 6. Put the model weights in place

The worker needs the *inference* weights only. Two options.

### Option A — copy from the Mac (fastest, recommended)

Over the network or a USB drive, copy these into the same relative paths on the
PC. Total ≈ 4.6 GB:

| From the Mac | Size | Needed for |
|---|---|---|
| `models/musetalkV15/` (`unet.pth` + `musetalk.json`) | 3.2 GB | MuseTalk |
| `models/sd-vae/` | 638 MB | MuseTalk |
| `models/wav2lip.pth` | 205 MB | Wav2Lip |
| `models/RealESRGAN_x4plus.pth` | 64 MB | enhancer |
| `models/GFPGANv1.4.pth` | 332 MB | enhancer |
| `gfpgan/weights/` (`detection_Resnet50_Final.pth` + `parsing_parsenet.pth`) | 195 MB | enhancer (face detect/parse) |
| `data/avatars/<id>/` for every avatar you stream | ~140 MB each | both |

The `gfpgan/weights/` entry is easy to miss: `facexlib` (used by GFPGAN) looks
for its face-detection and face-parsing weights in the **cwd-relative**
`gfpgan/weights/`, not in site-packages. They are in the repo on the Mac. If you
omit them, the worker downloads them on the first enhance job (~195 MB, once) —
not fatal, just slower.

Do **not** copy `models/whisper/` — that is the Mac's ASR model; feature
extraction stays on the Mac. Do **not** copy `models/musetalk/` (the v1
checkpoint) unless you actually use v1; v15 is what the code loads.

Practical shortcut with `scp` from the Mac (adjust host/paths):

```bash
# on the Mac
scp -r models/musetalkV15 models/sd-vae models/wav2lip.pth \
       models/RealESRGAN_x4plus.pth models/GFPGANv1.4.pth \
       user@PC:/C:/work/livetalking/models/
scp -r gfpgan/weights user@PC:/C:/work/livetalking/gfpgan/
scp -r data/avatars/anthony005 user@PC:/C:/work/livetalking/data/avatars/
```

### Option B — download on the PC

```powershell
python scripts\download_musetalk_weights.py   # vae, whisper, face-parse, syncnet, dwpose
bash scripts\download_musetalk_unet.sh       # 3.4 GB unet.pth, resumable curl
python scripts\download_enhance_weights.py   # Real-ESRGAN + GFPGAN
```

The GFPGAN face-detection/parsing weights (`gfpgan/weights/`) are fetched
automatically by `facexlib` on the first enhance job if they are absent — no
separate step, just a one-time ~195 MB download.

`unet.pth` is 3.4 GB; the curl script is deliberately used instead of the
HuggingFace xet client, which can balloon the download past 4 GB. If you have no
`bash` on Windows, run that script's `curl` command from PowerShell manually, or
use Option A.

`models/wav2lip.pth` is not covered by the downloaders — see `MODELS.md`
(rename `wav2lip256.pth` to `wav2lip.pth`).

---

## 7. Open the firewall for the worker port

The Mac connects to TCP **50051**. Windows blocks inbound by default, and the
symptom is a client-side hang, not a clear "connection refused".

```powershell
New-NetFirewallRule -DisplayName "LiveTalking GPU worker" `
  -Direction Inbound -Protocol TCP -LocalPort 50051 -Action Allow
```

Restrict it to the Mac's address if the network is not trusted:

```powershell
New-NetFirewallRule -DisplayName "LiveTalking GPU worker" `
  -Direction Inbound -Protocol TCP -LocalPort 50051 -Action Allow `
  -RemoteAddress 192.168.1.50
```

Find the PC's LAN address for the Mac's `--remote_worker` argument:

```powershell
ipconfig | Select-String "IPv4"
```

Also set the network profile to **Private** (Settings → Network → Properties),
otherwise Windows applies the stricter Public profile.

---

## 8. Start the worker

**Always launch from the repo root.** `avatars/musetalk/utils/__init__.py` does a
`sys.path.append` that can shadow the project's `utils` package if the working
directory is wrong; launching from the root keeps `import utils.logger`
resolving to the repo's package.

```powershell
cd C:\work\livetalking
conda activate livetalking
python -m remote.worker --host 0.0.0.0 --port 50051 --backend musetalk
```

Or double-click `scripts\start_worker.bat` (edit its `REPO`/`CONDA_ENV` at the
top first). The .bat invokes the environment's `python.exe` directly rather than
calling `conda activate`, and refuses to start if
`torch.cuda.is_available()` is `False`.

Expected output, ending in:

```
[worker] musetalk device=cuda
[worker] musetalk weights loaded
[worker] listening on 0.0.0.0:50051 (backend=musetalk)
```

Notes on the flags:

- `--host 0.0.0.0` is required for the Mac to reach it; `127.0.0.1` only accepts
  local connections.
- `--backend musetalk|wav2lip` preloads weights (slower startup, first request is
  fast). Omit it to load lazily on the Mac's first `LoadModel`.
- If startup dies with `ModuleNotFoundError: No module named 'utils.logger'`, you
  are in the wrong directory or launched `remote/worker.py` as a file path
  instead of with `-m`.
- First run downloads RVM's `rvm_mobilenetv3.pth` via `torch.hub` into
  `%USERPROFILE%\.cache\torch\hub\checkpoints\` (~15 MB).

---

## 9. Verify from the Mac

Before touching the app, prove the transport + GPU work:

```bash
cd /path/to/livetalking
/opt/miniconda3/envs/livetalking/bin/python -c \
  "from remote.client import probe; print(probe('192.168.1.42:50051'))"
```

Expected:

```
{'ok': True, 'device': 'cuda', 'backend': 'musetalk' or '', 'avatar_id': '', 'detail': 'NVIDIA GeForce RTX ...'}
```

`device: 'cuda'` plus the GPU name means the whole point of the exercise
worked. `device: 'cpu'` means stage 3 went wrong — go back and re-verify
`torch.cuda.is_available()` **inside the worker's own environment**.

Then a real inference round-trip with an avatar that exists on the PC:

```bash
/opt/miniconda3/envs/livetalking/bin/python -c "
import numpy as np
from remote.client import RemoteClient
c = RemoteClient('192.168.1.42:50051', timeout=600)
c.load_model('musetalk')
c.load_avatar('musetalk', 'anthony005', [])
print(c.infer_batch(0, 4, [np.zeros((50,384), np.uint8)]*4).shape)
"
```

Expected `(4, 256, 256, 3)`. Passing `[]` for the avatar frames exercises the
worker's on-disk path (stage 6); if that raises `FileNotFoundError`, the avatar
folder is missing or in the wrong place.

---

## 10. Run the app from the Mac against it

```bash
cd /path/to/livetalking
./run.sh --model remote --remote_backend musetalk \
         --remote_worker 192.168.1.42:50051 \
         --avatar_id anthony005
```

Then open `http://localhost:8010/index.html`. Confirm real-time from the server
log — both numbers must be ≥ 25:

```
------actual avg infer fps:31.2
```

The same `--remote_worker` also routes the web enhancer (RVM + Real-ESRGAN +
GFPGAN) to the PC; no extra config. Background-swap "recompose" stays local.

---

## Keeping it running

The worker is a plain foreground process — closing the Anaconda Prompt window
stops it. To run it as a background service:

- Simplest: a `start_worker.bat` on the Desktop that does
  `cd /d C:\work\livetalking && call conda activate livetalking && python -m remote.worker --host 0.0.0.0 --port 50051`,
  launched via Task Scheduler "At log on".
- Or register it with [NSSM](https://nssm.cc/) as a Windows service, pointing at
  `C:\...\envs\livetalking\python.exe` with args `-m remote.worker --host 0.0.0.0 --port 50051`
  and `C:\work\livetalking` as the working directory.

Prevent the GPU from idling to sleep mid-session: NVIDIA Control Panel → Manage
3D settings → Power management mode → **Prefer maximum performance**, and set
Windows power plan to High performance.

---

## Troubleshooting quick reference

| Symptom | Cause / fix |
|---|---|
| `torch.cuda.is_available()` is `False` | CPU wheel installed, or driver older than the wheel's CUDA. Stage 3 / stage 0. |
| Mac hangs connecting, no error | Firewall rule missing, or `--host 127.0.0.1`. Stage 7. |
| `ModuleNotFoundError: No module named 'utils.logger'` | Launched from the wrong directory or as a file path. Stage 8. |
| `torchvision.transforms.functional_tensor` on import | basicsr not patched. Stage 5. |
| `no model loaded; call LoadModel first` | Worker has no weights or backend name mismatch. Stage 6 / `--remote_backend`. |
| `LoadAvatar failed: ...latents.pt` | Avatar folder absent on the PC and no frames shipped. Stage 6, or let the Mac ship them (it does by default). |
| `CUDA out of memory` | Lower `--batch_size` on the Mac (8 → 4), or call `POST /api/model/unload` between sessions. |
| First batch very slow, later fast | Normal: weights load + warm-up on first session. |
| `resource_tracker: leaked semaphore objects` on Ctrl-C | Harmless on shutdown. |
| Worker log shows a `conda` traceback (`_get_deactivate_scripts`) | Your conda install's shell hook is broken, not the worker. Launch with the env's `python.exe` directly (as `start_worker.bat` does) and it never runs. |
