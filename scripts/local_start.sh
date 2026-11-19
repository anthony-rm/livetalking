#!/usr/bin/env bash
# Start the LOCAL dev server (this Mac — app only, no avatars, no weights).
#
# This Mac is not the avatar host: default avatars are the ones created for
# the REMOTE server. So this starts with --skip-model (no weights load at
# boot; state stays 'unloaded', lazy on connect) and binds 127.0.0.1:8010,
# reached from elsewhere over an SSH tunnel.
#
# Usage:  ./scripts/local_start.sh [--skip-model]
# Env:   LT_PYTHON — override the conda interpreter path
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

ARGS=("--skip-model")
# Forward any extra args (e.g. --listenport 8011) to the server.
[[ $# -gt 0 ]] && ARGS=("$@")

if [ -f livetalking.pid ] && kill -0 "$(cat livetalking.pid 2>/dev/null)" 2>/dev/null; then
    echo "local server already running (pid $(cat livetalking.pid))"
    exit 0
fi
# A stale pid file from a crashed process must not be trusted.
[ -f livetalking.pid ] && rm -f livetalking.pid

nohup ./restart.sh "${ARGS[@]}" > /tmp/livetalking_local.log 2>&1 &

# Wait for the pidfile the server writes itself, or bail out if it died.
deadline=$(( $(date +%s) + 20 ))
while [ ! -f livetalking.pid ] || ! kill -0 "$(cat livetalking.pid)" 2>/dev/null; do
    if [ $(date +%s) -ge $deadline ]; then
        echo "local server did not come up in 20s — see /tmp/livetalking_local.log" >&2
        exit 1
    fi
    sleep 0.3
done
PID="$(cat livetalking.pid)"
echo "local server started (pid $PID)"
curl -s -m 5 "http://127.0.0.1:${PORT:-8010}/api/model/status" || true