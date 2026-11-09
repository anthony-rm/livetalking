#!/usr/bin/env bash
# Restart the LiveTalking server cleanly.
#
# Usage:   ./restart.sh [same args you would pass to run.sh]
# Example: ./restart.sh --skip-model --listenport 8031 --transport webrtc
#
# What it does:
#   1. Reads the port out of --listenport (default 8031, also honours config.yaml
#      indirectly because app.py's default matches).
#   2. Stops the process recorded in livetalking.pid (SIGTERM, then SIGKILL after
#      a timeout) and any other python app.py still listening on that port.
#   3. Refuses to kill a non-app.py process holding the port.
#   4. Relaunches via run.sh with the args you passed.
#
# Env: LT_STOP_TIMEOUT (seconds to wait for a graceful exit, default 15)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/livetalking.pid"
STOP_TIMEOUT="${LT_STOP_TIMEOUT:-15}"
PORT=8031

# ── parse --listenport from our own args ─────────────────────────────────────
prev=""
for arg in "$@"; do
    case "$arg" in
        --listenport=*) PORT="${arg#*=}" ;;
    esac
    if [[ "$prev" == "--listenport" ]]; then
        PORT="$arg"
    fi
    prev="$arg"
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "restart.sh: could not parse port from --listenport ('$PORT')" >&2
    exit 2
fi

# ── collect candidate PIDs ──────────────────────────────────────────────────
pids=()
add_pid() {
    local pid="$1"
    for seen in "${pids[@]:-}"; do
        [[ "$seen" == "$pid" ]] && return
    done
    pids+=("$pid")
}

if [[ -f "$PID_FILE" ]]; then
    pidfile_pid="$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pidfile_pid:-}" ]] && kill -0 "$pidfile_pid" 2>/dev/null; then
        add_pid "$pidfile_pid"
    else
        rm -f "$PID_FILE"   # stale pid file
    fi
fi

blocked_pids=()
while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    cmd="$(ps -o command= -p "$pid" 2>/dev/null || true)"
    if [[ "$cmd" == *app.py* ]]; then
        add_pid "$pid"
    else
        blocked_pids+=("$pid")
    fi
done < <(lsof -nP -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null || true)

if [[ ${#blocked_pids[@]} -gt 0 ]]; then
    echo "restart.sh: port $PORT is held by a process that is not LiveTalking:" >&2
    for pid in "${blocked_pids[@]}"; do
        echo "  PID $pid  $(ps -o command= -p "$pid" 2>/dev/null || echo '<gone>')" >&2
    done
    echo "Not killing it. Free the port yourself or pick another --listenport." >&2
    exit 1
fi

# ── stop ────────────────────────────────────────────────────────────────────
if [[ ${#pids[@]} -eq 0 ]]; then
    echo "restart.sh: nothing running on port $PORT — starting fresh"
else
    echo "restart.sh: stopping PID(s) ${pids[*]} on port $PORT"
    kill -TERM "${pids[@]}" 2>/dev/null || true

    deadline=$(( $(date +%s) + STOP_TIMEOUT ))
    while [[ $(date +%s) -lt $deadline ]]; do
        alive=0
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        [[ $alive -eq 0 ]] && break
        sleep 0.3
    done

    survivors=()
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && survivors+=("$pid")
    done
    if [[ ${#survivors[@]} -gt 0 ]]; then
        echo "restart.sh: PID(s) ${survivors[*]} ignored SIGTERM after ${STOP_TIMEOUT}s — sending SIGKILL"
        kill -KILL "${survivors[@]}" 2>/dev/null || true
        sleep 0.5
    fi
fi

# Port must be free before we hand control to run.sh, otherwise the new process
# dies with "address already in use" again.
if lsof -nP -ti "tcp:$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "restart.sh: port $PORT is STILL in use after stopping:" >&2
    lsof -nP -i "tcp:$PORT" -sTCP:LISTEN >&2 || true
    exit 1
fi

rm -f "$PID_FILE"

# ── start ───────────────────────────────────────────────────────────────────
echo "restart.sh: starting $(printf '%q ' ./run.sh "$@")"
exec "$SCRIPT_DIR/run.sh" "$@"
