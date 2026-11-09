#!/usr/bin/env bash
# LiveTalking environment setup (macOS / Apple Silicon)
# Idempotent: re-running is safe.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="livetalking"
# Resolve the env prefix from conda itself so this works where miniconda lives
# outside /opt (Homebrew, miniforge, ...). LT_PYTHON overrides everything.
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
CONDA_PY="${LT_PYTHON:-${CONDA_BASE}/envs/${ENV_NAME}/bin/python}"
PIP="$(dirname "$CONDA_PY")/pip"

echo "[1/5] Conda env: ${ENV_NAME}"
if ! conda env list | grep -q "^\s*${ENV_NAME}\s"; then
  conda create -y -n "${ENV_NAME}" python=3.12
fi

echo "[2/5] Install Python deps"
# torch CPU/MPS build for Apple Silicon
"$PIP" install --upgrade pip
"$PIP" install torch torchvision torchaudio
"$PIP" install -r requirements.txt

# MuseTalk avatar generation needs dlib 68-point landmarks via face_recognition
# (avatars/musetalk/utils/preprocessing.py). Install the prebuilt dlib-bin wheel
# instead of compiling dlib from source, and skip face-recognition's own `dlib`
# dependency so pip does not try to build it. setuptools<81 is required because
# face_recognition_models imports the removed `pkg_resources` module.
"$PIP" install dlib-bin face_recognition_models "setuptools<81"
"$PIP" install --no-deps face-recognition

# gdown is needed by scripts/download_musetalk_weights.py for the face-parsing
# checkpoint, which is only distributed via Google Drive.
"$PIP" install gdown

# edge-tts 7.2.8 is broken against Bing's current TTS endpoint (403 on the
# WebSocket handshake). 7.3.1+ works. Pin it explicitly.
"$PIP" install --upgrade "edge-tts>=7.3.1"

# Avatar video preprocessing: gfpgan/realesrgan are in requirements.txt, but
# basicsr 1.4.2 (pulled transitively) imports torchvision.transforms.functional_tensor,
# which torchvision drops in 0.17+. Re-point it at functional; idempotent.
"$CONDA_PY" scripts/patch_basicsr.py

echo "[3/5] Copy .env"
cp -n .env.example .env 2>/dev/null || true

echo "[4/5] System deps"
# ffmpeg (already present via brew on this machine)
command -v ffmpeg >/dev/null 2>&1 || echo "  WARN: ffmpeg not found — install with: brew install ffmpeg"

echo "[5/5] Done."
echo ""
echo "Next steps:"
echo "  1. Download model weights (see MODELS.md):"
echo "     - models/wav2lip.pth  (rename from wav2lip256.pth)"
echo "     - data/avatars/wav2lip256_avatar1/  (from wav2lip256_avatar1.tar.gz)"
echo "  2. Avatar video preprocessing weights (tools/enhance, optional):"
echo "     - \$(conda run -n livetalking python scripts/download_enhance_weights.py) (or /opt/miniconda3/envs/livetalking/bin/python)"
echo "  3. Edit .env with your API keys (TENCENT / DASHSCOPE / AZURE / DOUBAO)"
echo "  4. Run:  ./run.sh --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1"
echo "  5. Open http://localhost:8010/index.html"