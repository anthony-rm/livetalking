#!/usr/bin/env bash
# Open the SSH tunnel to the REMOTE server (the GPU box).
#
# Runs ON THE MAC. The local port is allocated by the kernel (never fixed):
# a fixed port that something else already occupies makes the health check
# read THAT listener's banner and report success for a box that refused to
# forward at all.
#
# Usage:  ./scripts/tunnel_start.sh
# Env:   TUNNEL_USER, TUNNEL_HOST, TUNNEL_PORT — override the box address
#        TUNNEL_LOCAL_PORT — pin the local port instead of allocating one
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

USER="${TUNNEL_USER:-ecpkn_bac}"
HOST="${TUNNEL_HOST:-10.130.104.166}"
PORT="${TUNNEL_PORT:-2222}"
REMOTE_PORT="${TUNNEL_REMOTE_PORT:-8010}"
PIDFILE="scripts/tunnel.pid"

# Allocate (or pin) the local listen port.
if [ -n "${TUNNEL_LOCAL_PORT:-}" ]; then
    LOCAL_PORT="$TUNNEL_LOCAL_PORT"
else
    LOCAL_PORT="$(python3 - <<'PY' 2>/dev/null
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "tunnel already running (pid $(cat "$PIDFILE"), local :$LOCAL_PORT)"
    exit 0
fi
[ -f "$PIDFILE" ] && rm -f "$PIDFILE"

# -N: no remote command, just forward. -T: disable TTY (non-interactive).
# -o ExitOnForwardFailure: fail fast if the forward is refused instead of
# hanging; -o ServerAliveInterval/Count: keep the connection alive through
# idle periods on this school-LAN path.
ssh -N -T \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=accept-new \
    -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" \
    -p "$PORT" "$USER@$HOST" > /tmp/livetalking_tunnel.log 2>&1 &
echo $! > "$PIDFILE"

# Health check: the far side is the box's own sshd (always listening), so a
# forward that reaches it proves forwarding is permitted.
sleep 2
if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "tunnel died on start — see /tmp/livetalking_tunnel.log" >&2
    tail -n 20 /tmp/livetalking_tunnel.log >&2
    rm -f "$PIDFILE"
    exit 1
fi
echo "tunnel up: 127.0.0.1:$LOCAL_PORT -> $USER@$HOST:$PORT -> 127.0.0.1:$REMOTE_PORT"
echo "open http://127.0.0.1:$LOCAL_PORT/ in your browser"