# Model weights

The model checkpoints are **not** included in the git repo (too large).
Download them separately and place as described below.

## wav2lip (default, recommended first model)

| File | Where it goes | Source |
|------|---------------|--------|
| `wav2lip256.pth` | `models/wav2lip.pth` (rename) | Wav2Lip checkpoint — see the note below |
| `wav2lip256_avatar1.tar.gz` | extract to `data/avatars/wav2lip256_avatar1/` | Sample avatar bundle — see the note below |

The two bundled Wav2Lip artifacts for this course project are distributed by the
team out of band (shared drive / lab machine), so there is no public link here.
Any `wav2lip256` checkpoint works: drop it at `models/wav2lip.pth` and extract an
avatar bundle into `data/avatars/<avatar_id>/`.

```bash
cd /path/to/livetalking
mv models/wav2lip256.pth models/wav2lip.pth
mkdir -p data/avatars
tar -xzf wav2lip256_avatar1.tar.gz -C data/avatars/
```

## musetalk

MuseTalk needs several checkpoints that are **not** in this repo. The full list
and sources come from the upstream `download_weights.sh`
(`https://github.com/TMElyralab/MuseTalk`):

| File | Where it goes | Source |
|------|---------------|--------|
| `musetalkV15/musetalk.json` + `unet.pth` | `models/musetalkV15/` | HF `TMElyralab/MuseTalk` |
| `sd-vae/config.json` + `diffusion_pytorch_model.bin` | `models/sd-vae/` | HF `stabilityai/sd-vae-ft-mse` |
| `whisper/config.json` + `pytorch_model.bin` | `models/whisper/` | HF `openai/whisper-tiny` |
| `face-parse-bisent/79999_iter.pth` | `models/face-parse-bisent/` | Google Drive (`gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812`) |
| `face-parse-bisent/resnet18-5c106cde.pth` | `models/face-parse-bisent/` | https://download.pytorch.org/models/resnet18-5c106cde.pth |

The VAE is the one that fails first — `avatars/musetalk/utils/utils.py` calls
`AutoencoderKL.from_pretrained("models/sd-vae")`, and if that folder is absent
diffusers tries to resolve `sd-vae` as a model id and errors out.

```bash
python scripts/download_musetalk_weights.py        # v1 + v15 + vae + whisper + syncnet + dwpose
bash   scripts/download_musetalk_unet.sh           # 3.4 GB unet.pth via curl+resume (see below)
# face-parse-bisent/79999_iter.pth needs gdown (Google Drive) — handled by the
# python script above; setup.sh installs gdown.
```

`unet.pth` is 3.4 GB and LFS/xet-backed. The `huggingface_hub` xet client
splits it into byte-range chunks and can duplicate ranges, ballooning the cache
past 4 GB for a 3.4 GB file. `scripts/download_musetalk_unet.sh` uses plain
`curl -L -C -` against the resolve URL instead — single stream, resumable, and
it verifies the final byte count.

Note: `models/sd-vae/diffusion_pytorch_model.safetensors` is preferred over the
`.bin` variant — diffusers warns about unsafe pickle serialization otherwise.

## ultralight

- Place checkpoint inside `models/ultralight/` (see `avatars/ultralight/`).
- Avatar folders go in `data/avatars/`.

## TTS / LLM

No local weights required for the default `edgetts` TTS. LLM defaults to `dashscope`
(Alibaba Qwen) — set `DASHSCOPE_API_KEY=...` in `.env` to enable conversational mode.

## Verification

After placing weights, the server should start without a `FileNotFoundError`:

```bash
./run.sh --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
```

Then open `http://localhost:8010/index.html` in a browser.

### Automated tests

Two test scripts in `tests/` exercise the running server:

```bash
# HTTP/API layer only — no model weights needed (uses --skip-model)
/opt/miniconda3/envs/livetalking/bin/python tests/smoke_test.py --port 8010

# Full end-to-end: real WebRTC handshake + text-driven speech
/opt/miniconda3/envs/livetalking/bin/python tests/webrtc_smoke.py --port 8010
```

Both should report 8/8 checks passed.

### Known gotcha

`edge-tts==7.2.8` (the version pinned in `requirements.txt`) is broken — Bing's
TTS endpoint now rejects its WebSocket handshake with a 403. The avatar
accepts `/human` but never speaks. Upgrade it:

```bash
pip install --upgrade "edge-tts>=7.3.1"
```