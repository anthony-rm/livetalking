#!/usr/bin/env bash
# Diagnose SSH-only connectivity to the GPU box, ONE cause at a time.
#
# Why this exists: a single `ssh -L` + `curl` test conflates four independent
# failures, and all four look like "curl prints 000":
#
#   1. no SSH key / auth rejected          -> nothing to tunnel
#   2. TCP forwarding disabled on the box  -> "administratively prohibited"
#   3. the tunnel opened but the far side has no listener -> empty reply
#   4. the worker is listening somewhere else (wrong port / 0.0.0.0 vs 127.0.0.1)
#
# So the tests are ordered cheapest-and-most-fundamental first, and each one
# names its own cause. Test B forwards to the box's OWN sshd (127.0.0.1:22),
# which is always listening — if that works, forwarding is permitted and any
# later failure is about the worker, not about SSH.
#
# Usage:  ./scripts/check_ssh_forwarding.sh [user@host] [worker_port]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

BOX="${1:-ecpkn_bac@10.130.105.203}"
WPORT="${2:-50051}"

# Local ports for the two probe forwards. These MUST be allocated, never fixed:
# a fixed port that something else already occupies makes section B read THAT
# listener's banner and report "forwarding works" for a box that refuses to
# forward at all — a false PASS on the one test whose whole job is to detect a
# refusal. A busy port also turns a healthy box into "could not forward".
free_port() {
  local p
  # Bind :0, read the kernel-chosen port, release it. Race-prone in theory
  # (another process could grab it in the microsecond before ssh binds) but
  # vastly better than a constant: the window is tiny, the fixed-port failure
  # was permanent.
  p=$(python3 - <<'PY' 2>/dev/null
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
  # Fallback ladder when python3 is unavailable.
  if [[ -z "$p" ]]; then
    for p in $(seq 60000 61000); do
      (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null || { echo "$p"; break; }
      exec 3>&- 2>/dev/null
    done
  fi
  echo "$p"
}

LP_A=$(free_port)   # local port for the sshd-banner test
LP_B=$(free_port)   # local port for the worker test
while [[ "$LP_B" == "$LP_A" ]]; do LP_B=$(free_port); done
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ExitOnForwardFailure=yes)

ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

cleanup() {
  [[ -n "${PID_A:-}" ]] && kill "$PID_A" 2>/dev/null
  [[ -n "${PID_B:-}" ]] && kill "$PID_B" 2>/dev/null
  wait 2>/dev/null
  return 0
}
trap cleanup EXIT

# Read the first bytes a TCP peer sends. Used to grab an SSH banner.
peek() {
  local host="$1" port="$2"
  if command -v nc >/dev/null 2>&1; then
    nc -w 3 "$host" "$port" 2>/dev/null | head -c 60
  else
    (exec 3<>"/dev/tcp/$host/$port" && head -c 60 <&3) 2>/dev/null
  fi
}

wait_for_port() {
  local port="$1" tries="${2:-40}"
  for _ in $(seq 1 "$tries"); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      exec 3>&- 2>/dev/null
      return 0
    fi
    sleep 0.25
  done
  return 1
}

ERR=$(mktemp -t lt_fwd_err.XXXXXX)
trap 'cleanup; rm -f "$ERR"' EXIT

# `timeout` is GNU coreutils; macOS does not ship it. Fall back to plain ssh
# (ConnectTimeout already bounds the connect phase) so this runs on the host.
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD=(timeout 25)
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD=(gtimeout 25)
else
  TIMEOUT_CMD=()
fi

head_ "A. SSH login to $BOX"
if "${TIMEOUT_CMD[@]+"${TIMEOUT_CMD[@]}"}" ssh "${SSH_OPTS[@]}" "$BOX" true 2>"$ERR"; then
  ok "key-based login works"
else
  bad "cannot log in with a key"
  sed 's/^/    /' "$ERR" | head -5
  note "No tunnel can work without this. Install the key first:"
  note "    ./deploy.sh --print-key     # then paste the block on the box"
  exit 1
fi

head_ "B. Is TCP forwarding permitted?  (forwards to the box's own sshd on :22)"
ssh "${SSH_OPTS[@]}" -N -L "127.0.0.1:$LP_A:127.0.0.1:22" "$BOX" 2>"$ERR" &
PID_A=$!
if ! wait_for_port "$LP_A"; then
  bad "the local forward never opened"
  sed 's/^/    /' "$ERR" | head -8
  if grep -qi 'administratively prohibited' "$ERR"; then
    note "TCP forwarding is DISABLED on the server. Only its administrator can"
    note "enable it: put 'AllowTcpForwarding local' in /etc/ssh/sshd_config and"
    note "reload sshd. This is a permission to forward, not an open port."
  elif grep -qi 'cannot listen\|address already in use' "$ERR"; then
    note "Local port $LP_A is busy on THIS machine — close that other tunnel."
  fi
  exit 1
fi

BANNER=$(peek 127.0.0.1 "$LP_A")
if [[ "$BANNER" == SSH-* ]]; then
  ok "forwarding works: reached the remote sshd ($(echo "$BANNER" | tr -d '\r' | cut -c1-40))"
else
  bad "tunnel opened but the far end sent no SSH banner (got: '${BANNER:-<empty>}')"
  note "Something odd about the forward; check: ssh -vvv -N -L ... $BOX"
  exit 1
fi
kill "$PID_A" 2>/dev/null; PID_A=

head_ "C. Is the worker listening on the box?"
# Emit BOTH the local address and the peer column explicitly, so the exposure
# check reads a field instead of pattern-matching a whole line: `ss -ltn` pads
# the peer address as '*:*' on every row, so grepping the line for '*:' matched
# a loopback-bound worker and warned that it was exposed to the whole network.
# Field 4 is Local Address:Port, which is the only thing that matters here.
REMOTE_LISTEN=$(ssh "${SSH_OPTS[@]}" "$BOX" \
  "ss -ltn 2>/dev/null | awk '\$4 ~ /:$WPORT\$/ {print \$4}' || netstat -ltn 2>/dev/null | awk '\$4 ~ /:$WPORT\$/ {print \$4}'")
if [[ -n "$REMOTE_LISTEN" ]]; then
  ok "something is listening on :$WPORT"
  echo "$REMOTE_LISTEN" | sed 's/^/      /'
  # Exposure test on the LOCAL ADDRESS ONLY. Loopback forms are fine:
  #   127.0.0.1:P    [::1]:P    [::ffff:127.0.0.1]:P
  # Anything else (0.0.0.0:P, [::]:P, a LAN IP) is reachable off-box.
  EXPOSED=""
  while read -r addr; do
    [[ -z "$addr" ]] && continue
    case "$addr" in
      127.0.0.1:*|\[::1\]:*|\[::ffff:127.0.0.1\]:*) : ;;
      *) EXPOSED="$EXPOSED $addr" ;;
    esac
  done <<< "$REMOTE_LISTEN"
  if [[ -n "$EXPOSED" ]]; then
    note "WARNING: reachable off-box on:$EXPOSED"
    note "That exposes the unauthenticated gRPC port to the whole network."
    note "Restart it via ./deploy.sh so it binds 127.0.0.1."
  fi
else
  bad "nothing is listening on :$WPORT on the box"
  note "The worker is not running. On the box:"
  note "    cd ~/livetalking/current"
  note "    PY=\$(cat ~/livetalking/.deploy/python)"
  note "    \$PY -m remote.worker --host 127.0.0.1 --port $WPORT"
  note "Or from here, once the code is deployed:  ./deploy.sh"
  exit 1
fi

head_ "D. gRPC health through a tunnel"
ssh "${SSH_OPTS[@]}" -N -L "127.0.0.1:$LP_B:127.0.0.1:$WPORT" "$BOX" 2>"$ERR" &
PID_B=$!
if ! wait_for_port "$LP_B"; then
  bad "could not forward 127.0.0.1:$LP_B -> 127.0.0.1:$WPORT"
  sed 's/^/    /' "$ERR" | head -8
  exit 1
fi

PY="${LT_PYTHON:-/opt/miniconda3/envs/livetalking/bin/python}"
[[ -x "$PY" ]] || PY=$(command -v python3)
if "$PY" - <<PY 2>/dev/null
import sys
sys.path.insert(0, ".")
from remote.client import probe
info = probe("127.0.0.1:$LP_B", timeout=8)
print("      device=%s backend=%s revision=%s gpu=%s" % (
    info.get("device"), info.get("backend"), info.get("revision"), info.get("detail")))
raise SystemExit(0 if info.get("ok") else 1)
PY
then
  ok "worker answered Health through the tunnel"
else
  bad "port is open inside the box but no gRPC Health reply"
  note "Something is listening on :$WPORT that is not the worker."
  exit 1
fi

head_ "Result"
ok "SSH-only transport to $BOX is fully working"
note "Nothing else to do: ./deploy.sh will use exactly this path."
exit 0
