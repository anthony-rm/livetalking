#!/usr/bin/env bash
# Start the REMOTE server (the GPU box — where the avatars live).
#
# Runs ON THE MAC. The box-side logic is piped to the box over SSH via
# 'bash -s' stdin (never embed shell in the ssh argument string — nested
# quoting parses wrong remotely). An interactive password prompt, if any,
# still works: ssh reads it from the TTY.
#
# Usage: ./scripts/remote_start.sh
# Env:  TUNNEL_USER / TUNNEL_HOST / TUNNEL_PORT — box address overrides
set -uo pipefail

RUSER="${TUNNEL_USER:-ecpkn_bac}"
HOST="${TUNNEL_HOST:-10.130.104.166}"
SPORT="${TUNNEL_PORT:-2222}"

ssh -p "$SPORT" "$RUSER@$HOST" 'bash -s' <<'REMOTE'
set -uo pipefail
REPO="$HOME/livetalking_box"
PORT=8010

cd "$REPO" || { echo "repo not found at $REPO" >&2; exit 1; }

if [ -f livetalking.pid ] && kill -0 "$(cat livetalking.pid 2>/dev/null)" 2>/dev/null; then
    echo "remote server already running (pid $(cat livetalking.pid))"
    exit 0
fi

# LT_PYTHON must be exported: box run.sh hardcodes /opt/miniconda3/... while
# the env lives under $HOME — without it restart.sh kills the running server
# and then dies on "interpreter not found", leaving the port dead.
export LT_PYTHON="${LT_PYTHON:-$HOME/miniconda3/envs/livetalking/bin/python}"
[ -x "$LT_PYTHON" ] || { echo "interpreter not found: $LT_PYTHON" >&2; exit 1; }

# setsid detaches the restart chain so ssh can close cleanly; </dev/null and
# the redirections stop the session hanging on the child's stdio.
setsid nohup ./restart.sh </dev/null >/tmp/livetalking_box.log 2>&1 &

# The pidfile appears only after app.py's import phase (torch etc.) — allow
# 60s.
deadline=$(( $(date +%s) + 60 ))
while [ $(date +%s) -lt $deadline ]; do
    if [ -f livetalking.pid ] && kill -0 "$(cat livetalking.pid 2>/dev/null)" 2>/dev/null; then
        echo "remote server started (pid $(cat livetalking.pid))"
        echo -n "state: "
        curl -s -m 5 "http://127.0.0.1:$PORT/api/model/status" \
            || echo "(http not bound yet — give it a few seconds)"
        echo
        exit 0
    fi
    sleep 0.5
done
echo "remote server did not come up in 60s — box log tail:" >&2
tail -n 15 /tmp/livetalking_box.log >&2
exit 1
REMOTE