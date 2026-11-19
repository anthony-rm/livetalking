#!/usr/bin/env bash
# Stop the REMOTE server (the GPU box).
#
# Runs ON THE MAC; the box-side stop logic is piped over SSH via 'bash -s'
# stdin. SIGTERM first, SIGKILL after a timeout, and never kills a process
# that is not LiveTalking.
#
# Usage: ./scripts/remote_stop.sh
# Env:  TUNNEL_USER / TUNNEL_HOST / TUNNEL_PORT — box address overrides
set -uo pipefail

RUSER="${TUNNEL_USER:-ecpkn_bac}"
HOST="${TUNNEL_HOST:-10.130.104.166}"
SPORT="${TUNNEL_PORT:-2222}"

ssh -p "$SPORT" "$RUSER@$HOST" 'bash -s' <<'REMOTE'
set -uo pipefail
REPO="$HOME/livetalking_box"
PORT=8010
STOP_TIMEOUT="${LT_STOP_TIMEOUT:-15}"

cd "$REPO" || { echo "repo not found at $REPO" >&2; exit 1; }

pids=()
if [ -f livetalking.pid ]; then
    pidfile_pid="$(tr -dc '0-9' < livetalking.pid 2>/dev/null || true)"
    if [ -n "${pidfile_pid:-}" ] && kill -0 "$pidfile_pid" 2>/dev/null; then
        pids+=("$pidfile_pid")
    else
        rm -f livetalking.pid
    fi
fi
# Anyone else running app.py on the app port.
while IFS= read -r pid; do
    [ -z "$pid" ] && continue
    cmd="$(ps -o command= -p "$pid" 2>/dev/null || true)"
    [[ "$cmd" == *app.py* ]] && pids+=("$pid")
done < <(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | sort -u)

if [ ${#pids[@]} -eq 0 ]; then
    echo "remote server not running"
    rm -f livetalking.pid
    exit 0
fi

echo "stopping remote server pid(s) ${pids[*]}"
kill -TERM "${pids[@]}" 2>/dev/null || true
deadline=$(( $(date +%s) + STOP_TIMEOUT ))
while [ $(date +%s) -lt $deadline ]; do
    alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && break
    sleep 0.3
done
for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && { echo "SIGKILL $pid"; kill -KILL "$pid" 2>/dev/null || true; }
done
sleep 0.5
rm -f livetalking.pid
echo "remote server stopped"
ss -ltn 2>/dev/null | grep -q ":$PORT " && echo "WARNING: port $PORT still held" || echo "port $PORT free"
REMOTE