#!/usr/bin/env bash
# Finish the two remaining weight files on the box, detached from any SSH session.
#
# Why detached: a long ssh session gets killed when the link hiccups, which is
# how the earlier attempts lost their progress mid-file. nohup + a loop that
# retries forever means the transfer resumes whenever bandwidth is available
# (the box is shared; another account's job can saturate the uplink for hours).
#
# Both files are idempotent and hash-verified, so running this repeatedly is safe.
set -uo pipefail

ROOT="$HOME/livetalking"
M="$ROOT/models"
LOG="$ROOT/logs/weights.log"
EXPECT_UNET=3400074924
MD5_UNET=c0b7bd9ae53dd3059495bc3058e62cad
MIRROR=https://hf-mirror.com

mkdir -p "$M/musetalkV15" "$(dirname "$LOG")"
log() { printf '%s [weights] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

size_of() { stat -c%s "$1" 2>/dev/null || echo 0; }

# ── unet: 3.4 GB ──────────────────────────────────────────────────────────
fetch_unet() {
  local dest="$M/musetalkV15/unet.pth"
  if [ "$(size_of "$dest")" = "$EXPECT_UNET" ] && \
     [ "$(md5sum "$dest" | cut -d' ' -f1)" = "$MD5_UNET" ]; then
    log "unet already complete and verified"; return 0
  fi
  log "unet: resuming from $(size_of "$dest") / $EXPECT_UNET"
  aria2c -c -x 16 -s 16 -k 1M --retry-wait=10 --max-tries=0 --timeout=30 \
    --connect-timeout=20 --file-allocation=none --allow-overwrite=false \
    --auto-file-renaming=false --summary-interval=120 --console-log-level=warn \
    -d "$M/musetalkV15" -o unet.pth \
    "$MIRROR/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth" >> "$LOG" 2>&1
  local got; got=$(size_of "$dest")
  if [ "$got" != "$EXPECT_UNET" ]; then
    log "unet: incomplete ($got)"; return 1
  fi
  local md5; md5=$(md5sum "$dest" | cut -d' ' -f1)
  if [ "$md5" != "$MD5_UNET" ]; then
    # A corrupt 3.4 GB file loads and then fails at inference; never keep it.
    log "unet: MD5 MISMATCH ($md5) -- deleting and retrying from scratch"
    rm -f "$dest" "$dest.aria2"
    return 1
  fi
  log "unet: verified OK"; return 0
}

# ── wav2lip: 205 MB, optional backend ─────────────────────────────────────
fetch_wav2lip() {
  local dest="$M/wav2lip.pth"
  [ "$(size_of "$dest")" -gt 200000000 ] && { log "wav2lip already present"; return 0; }
  # The camenduru mirror serves a 167 KB LFS POINTER, not the weights; these two
  # repos carry the real 435 MB checkpoint, which is the one MuseTalk/Wav2Lip
  # tooling in this repo tolerates (load_state_dict ignores the extra keys).
  for url in \
    "$MIRROR/Nekochu/Wav2Lip/resolve/main/wav2lip.pth" \
    "$MIRROR/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip.pth"; do
    log "wav2lip: trying $url"
    aria2c -c -x 8 -s 8 -k 1M --retry-wait=10 --max-tries=5 --timeout=30 \
      --connect-timeout=20 --file-allocation=none --allow-overwrite=true \
      --auto-file-renaming=false --summary-interval=120 --console-log-level=warn \
      -d "$M" -o wav2lip.pth "$url" >> "$LOG" 2>&1
    if [ "$(size_of "$dest")" -gt 400000000 ]; then
      log "wav2lip: downloaded ($(size_of "$dest") bytes)"; return 0
    fi
  done
  log "wav2lip: could not fetch a full checkpoint"; return 1
}

# Retry loop: the uplink is shared and can be saturated for hours by another
# account. Give up only after both succeed or the attempt budget is spent.
for attempt in $(seq 1 40); do
  ok=0
  fetch_unet && ok=$((ok+1))
  fetch_wav2lip && ok=$((ok+1))
  [ "$ok" -eq 2 ] && { log "ALL WEIGHTS COMPLETE"; exit 0; }
  log "attempt $attempt: $ok/2 done, sleeping 120s"
  sleep 120
done
log "gave up after 40 attempts"
exit 1
