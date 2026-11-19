#!/usr/bin/env bash
# Stop the SSH tunnel to the REMOTE server.
#
# Runs ON THE MAC. SIGTERM first (ssh -N honours it), SIGKILL after a
# timeout — never kill a process that is not our tunnel.
#
# Usage:  ./scripts/tunnel_stop.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
PIDFILE="scripts/tunnel.pid"
STOP_TIMEOUT="${LT_STOP_TIMEOUT:-10}"

if [ ! -f "$PIDFILE" ]; then
    echo "tunnel not running"
    exit 0
fi
PID="$(cat "$PIDFILE" 2>/dev/null)"
if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PIDFILE"
    echo "tunnel not running (stale pidfile removed)"
    exit 0
fi

echo "stopping tunnel pid $PID"
kill -TERM "$PID" 2>/dev/null || true
deadline=$(( $(date +%s) + STOP_TIMEOUT ))
while kill -0 "$PID" 2>/dev/null && [ $(date +%s) -lt $deadline ]; do
    sleep 0.3
done
kill -0 "$PID" 2>/dev/null && { echo "SIGKILL $PID"; kill -KILL "$PID" 2>/dev/null || true; }
rm -f "$PIDFILE"
echo "tunnel stopped"