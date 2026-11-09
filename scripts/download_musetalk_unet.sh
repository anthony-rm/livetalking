#!/usr/bin/env bash
# Download MuseTalk V1.5 unet.pth (3.4 GB, LFS/xet-backed) with curl + resume.
#
# Why not huggingface_hub: the xet client splits the file into byte-range
# chunks that are written as separate .incomplete files and sometimes duplicated
# ranges, ballooning the cache to >4 GB for a 3.4 GB file. Plain HTTP range
# requests against the resolve URL are single-stream and resumable.
#
# The HF resolve URL 302-redirects to a signed CDN URL (us.aws.cdn.hf.co) that
# honours Range requests, so `curl -L -C -` works and can resume across runs.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models/musetalkV15/unet.pth"
URL="https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth"
EXPECTED=3400074924

mkdir -p "$(dirname "$DEST")"

if [ -f "$DEST" ]; then
  SIZE=$(stat -f%z "$DEST" 2>/dev/null || stat -c%s "$DEST")
  if [ "$SIZE" -eq "$EXPECTED" ]; then
    echo "already complete: $DEST ($SIZE bytes)"
    exit 0
  fi
  echo "resuming from $SIZE / $EXPECTED"
fi

# --retry with --retry-all-errors so transient CDN 5xx / resets resume rather
# than aborting. -C - resumes a partial file.
curl -L -C - \
  --retry 10 --retry-delay 5 --retry-all-errors \
  --connect-timeout 30 \
  -o "$DEST" "$URL"

SIZE=$(stat -f%z "$DEST" 2>/dev/null || stat -c%s "$DEST")
if [ "$SIZE" -ne "$EXPECTED" ]; then
  echo "ERROR: size mismatch: got $SIZE, expected $EXPECTED" >&2
  exit 1
fi
echo "OK $DEST ($SIZE bytes)"
