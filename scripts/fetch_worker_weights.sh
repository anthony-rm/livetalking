#!/usr/bin/env bash
# Fetch ONLY the weights the worker actually loads, on the box, from CN mirrors.
#
# Rationale: huggingface.co gives ~105 KB/s from this box, hf-mirror.com ~380
# KB/s single-stream and ~1 MB/s with aria2c -x16. The SSH link caps at ~235
# KB/s, so downloading here beats transferring from the host.
#
# What is included — and why each is required:
#   models/musetalkV15/{unet.pth,musetalk.json}   inference (config.yaml: v15)
#   models/sd-vae/{config.json,diffusion_pytorch_model.bin}  VAE for infer AND genavatar
#   models/face-parse-bisent/*                    genavatar face parsing
#   models/dwpose/*                               genavatar pose
#   models/GFPGANv1.4.pth, RealESRGAN_x4plus.pth  video enhance
#   gfpgan/weights/*                              facexlib face detect/parse (enhance)
#   models/wav2lip.pth                            wav2lip inference (optional backend)
#
# Deliberately EXCLUDED:
#   models/musetalk/{pytorch_model.bin}  MuseTalk v1 — config.yaml uses v15
#   models/whisper/                      host-only; the worker no longer loads it
#                                        (audio features arrive pre-computed)
#   models/syncnet/                      training-time only
set -uo pipefail

ROOT="${LT_ROOT:-$HOME/livetalking}"
MODELS="$ROOT/models"
GFPGAN_W="$ROOT/gfpgan/weights"
HF="${HF_MIRROR:-https://hf-mirror.com}"
GH="${GH_PROXY:-https://ghproxy.net/https://github.com}"
JOBS="${JOBS:-16}"

mkdir -p "$MODELS"/{musetalkV15,sd-vae,face-parse-bisent,dwpose} "$GFPGAN_W"

log() { printf '[weights] %s\n' "$*"; }

# aria2c resumes via its own .aria2 control file. NEVER interrupt a live aria2c
# with a second writer (curl -C -) on the same path: the two disagree about the
# file map and the result is a corrupted or shrinking file. One writer at a time.
fetch() {
  local url="$1" dest="$2" expect="${3:-}"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && [ ! -f "$dest.aria2" ]; then
    local size; size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ -n "$expect" ]; then
      [ "$size" = "$expect" ] && { log "skip (complete) $(basename "$dest")"; return 0; }
    elif [ "$size" -gt 1024 ]; then
      log "skip (exists) $(basename "$dest")"; return 0
    fi
  fi
  log "-> $(basename "$dest") ${expect:+($expect bytes)}"
  aria2c -c -x "$JOBS" -s "$JOBS" -k 1M \
    --retry-wait=5 --max-tries=30 --timeout=30 --connect-timeout=20 \
    --file-allocation=none --allow-overwrite=false --auto-file-renaming=false \
    --summary-interval=15 \
    -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url" || true
  if [ ! -s "$dest" ] || [ -f "$dest.aria2" ]; then
    log "INCOMPLETE $(basename "$dest")"
    return 1
  fi
  if [ -n "$expect" ]; then
    local got; got=$(stat -c%s "$dest")
    [ "$got" = "$expect" ] || { log "SIZE MISMATCH got=$got want=$expect"; return 1; }
  fi
  log "done $(basename "$dest")"
}

MUSE="$HF/TMElyralab/MuseTalk/resolve/main"
SDVAE="$HF/stabilityai/sd-vae-ft-mse/resolve/main"

log "=== MuseTalk v1.5 inference"
fetch "$MUSE/musetalkV15/unet.pth"      "$MODELS/musetalkV15/unet.pth" 3400074924
fetch "$MUSE/musetalkV15/musetalk.json" "$MODELS/musetalkV15/musetalk.json"

log "=== SD VAE (shared: inference + genavatar)"
fetch "$SDVAE/config.json"                 "$MODELS/sd-vae/config.json"
fetch "$SDVAE/diffusion_pytorch_model.bin" "$MODELS/sd-vae/diffusion_pytorch_model.bin"

log "=== Wav2Lip (optional backend)"
fetch "$HF/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip.pth" "$MODELS/wav2lip.pth"

log "=== genavatar: pose + face parsing"
fetch "$HF/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" "$MODELS/dwpose/dw-ll_ucoco_384.pth"
fetch "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
      "$MODELS/face-parse-bisent/resnet18-5c106cde.pth"
if [ ! -s "$MODELS/face-parse-bisent/79999_iter.pth" ]; then
  log "-> 79999_iter.pth (Google Drive, via gdown)"
  LT_BISENET_OUT="$MODELS/face-parse-bisent/79999_iter.pth" \
  "${LT_PYTHON:-python3}" - <<'PYEOF'
import os, gdown
gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812",
               output=os.environ["LT_BISENET_OUT"], quiet=False)
PYEOF
fi

log "=== video enhance"
fetch "$GH/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
      "$MODELS/RealESRGAN_x4plus.pth"
fetch "$GH/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
      "$MODELS/GFPGANv1.4.pth"
# facexlib pins these tags in its own source; a wrong tag is a 404 at download.
fetch "$GH/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" \
      "$GFPGAN_W/detection_Resnet50_Final.pth"
fetch "$GH/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" \
      "$GFPGAN_W/parsing_parsenet.pth"

log "=== summary"
for f in "$MODELS"/musetalkV15/unet.pth "$MODELS"/sd-vae/diffusion_pytorch_model.bin \
         "$MODELS"/wav2lip.pth "$MODELS"/dwpose/dw-ll_ucoco_384.pth \
         "$MODELS"/face-parse-bisent/79999_iter.pth \
         "$MODELS"/GFPGANv1.4.pth "$MODELS"/RealESRGAN_x4plus.pth \
         "$GFPGAN_W"/detection_Resnet50_Final.pth "$GFPGAN_W"/parsing_parsenet.pth; do
  if [ -s "$f" ]; then printf '  OK   %8s  %s\n' "$(du -h "$f" | cut -f1)" "${f#$ROOT/}"
  else printf '  MISS %8s  %s\n' "-" "${f#$ROOT/}"; fi
done
