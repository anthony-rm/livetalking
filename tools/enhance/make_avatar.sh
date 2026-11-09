#!/usr/bin/env bash
# Enhance a raw talking-head clip and create a LiveTalking avatar from it.
# Usage:
#   ./tools/enhance/make_avatar.sh <raw_video> <avatar_id> [--model wav2lip|musetalk] [extra enhance flags]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO"

PY="${LT_PYTHON:-/opt/miniconda3/envs/livetalking/bin/python}"
RAW="$1"; shift
AVATAR_ID="$1"; shift
MODEL="${MODEL:-wav2lip}"

ENH="/tmp/${AVATAR_ID}_enhanced.mp4"
MATTE="/tmp/${AVATAR_ID}_matte"
PERSON="/tmp/${AVATAR_ID}_person"

echo "[1/3] enhancing $RAW -> $ENH (RVM matte + lighting + GFPGAN face restore)"
"$PY" -m tools.enhance.cli -i "$RAW" -o "$ENH" \
    --matte-out "$MATTE" --person-out "$PERSON" \
    --width 1080 --height 1920 --background '#1E1E1E' "$@"

echo "[2/3] generating avatar artifacts ($MODEL)"
case "$MODEL" in
  wav2lip)
    "$PY" -m avatars.wav2lip.genavatar --video_path "$ENH" \
        --avatar_id "$AVATAR_ID" --save_path data/avatars ;;
  musetalk)
    "$PY" -m avatars.musetalk.genavatar --file "$ENH" \
        --avatar_id "$AVATAR_ID" --save_path data/avatars ;;
  *)
    echo "unknown MODEL=$MODEL (use wav2lip|musetalk)"; exit 2 ;;
esac

echo "[3/3] done."
echo "  avatar : data/avatars/$AVATAR_ID/"
echo "  matte  : $MATTE (+ meta.json, person layer in $PERSON)"
echo "  swap bg: $PY -m tools.enhance.cli -i $RAW -o <out>.mp4 \\"
echo "             --background '#RRGGBB' --recompose-from $MATTE --person-out $PERSON"
echo "  run    : ./run.sh --transport webrtc --model $MODEL --avatar_id $AVATAR_ID"
