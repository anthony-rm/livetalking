#!/usr/bin/env bash
# Stop the LOCAL dev server (this Mac).
#
# Mirrors restart.sh's stop rules: SIGTERM first, SIGKILL after a timeout,
# and never kill a process that is not LiveTalking.
#
# Usage:  ./scripts/local_stop.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PORT=8010
STOP_TIMEOUT="${LT_STOP_TIMEOUT:-15}"
pids=()

if [ -f livetalking.pid ]; then
    pidfile_pid="$(tr -dc '0-9' < livetalking.pid 2>/dev/null || true)"
    if [ -n "${pidfile_pid:-}" ] && kill -0 "$pidfile_pid" 2>/dev/null; then
        pids+=("$pidfile_pid")
    else
        rm -f livetalking.pid   # stale
    fi
fi

# Any other python app.py on the port.
while IFS= read -r pid; do
    [ -z "$pid" ] && continue
    cmd="$(ps -o command= -p "$pid" 2>/dev/null || true)"
    if [[ "$cmd" == *app.py* ]]; then
        already=0
        for seen in "${pids[@]:-}"; do
            [ "$seen" == "$pid" ] && already=1
        done
        [ $already -eq 0 ] && pids+=("$pid")
    fi
done < <(lsof -nP -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null || true)

if [ ${#pids[@]} -eq 0 ]; then
    echo "local server not running"
    rm -f livetalking.pid
    exit 0
fi

echo "stopping local server pid(s) ${pids[*]}"
kill -TERM "${pids[@]}" 2>/dev/null || true
deadline=$(( $(date +%s) + STOP_TIMEOUT ))
while [ $(date +%s) -lt $deadline ]; do
    alive=0
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive=1
    done
    [ $alive -eq 0 ] && break
    sleep 0.3
done
for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && { echo "SIGKILL $pid"; kill -KILL "$pid" 2>/dev/null || true; }
done
sleep 0.5
rm -f livetalking.pid
echo "local server stopped"
lsof -nP -i "tcp:$PORT" -sTCP:LISTEN 2>/dev/null && echo "WARNING: port still held" || echo "port $PORT free"