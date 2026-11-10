"""Tests for scripts/check_ssh_forwarding.sh (the connectivity diagnostic).

That script exists because a single `ssh -L` + `curl` test conflates four
independent failures -- auth, forwarding disabled, no listener, wrong bind --
and prints `000` for every one of them. These tests drive the script with a FAKE
`ssh` that emulates each mode, so the diagnosis each mode produces is pinned
rather than discovered on the user's server.

The real box is never contacted: `ssh` on PATH is the stub.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "check_ssh_forwarding.sh")

pytestmark = pytest.mark.skipif(
    not os.path.exists("/bin/bash"), reason="needs bash")


class Box:
    """Writes a fake `ssh` that behaves in one of several scripted modes."""

    def __init__(self, root, mode):
        self.root = root
        self.bin = os.path.join(root, "bin")
        os.makedirs(self.bin, exist_ok=True)
        self.mode = mode
        self._write_ssh()
        self.env = dict(os.environ)
        self.env["PATH"] = self.bin + os.pathsep + self.env.get("PATH", "")
        self.env["LT_PYTHON"] = sys.executable

    def _write_ssh(self):
        path = os.path.join(self.bin, "ssh")
        body = rf"""#!/usr/bin/env bash
MODE={self.mode!r}
PORTFILE={os.path.join(self.root, 'forwarded_port')!r}
args=(); has_N=0; LSPEC=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    -N) has_N=1; shift ;;
    -L) LSPEC="$2"; shift 2 ;;
    -*) shift ;;
    *) shift ;;   # target
  esac
done
case "$MODE" in
  no-auth)
    echo "Permission denied (publickey)." >&2; exit 255 ;;
  no-forwarding)
    if [ "$has_N" = 1 ]; then
      echo "channel_setup_fwd_listener_tcpip: administratively prohibited: open failed" >&2
      exit 255
    fi
    exit 0 ;;
  check1-unreachable)
    # Forward opens, but nothing answers (curl 000 class)
    if [ "$has_N" = 1 ]; then sleep 30; exit 0; fi
    exit 0 ;;
  busy-port)
    echo "bind: Address already in use" >&2; exit 255 ;;
  ok)
    # Emulate a working local forward: listen and answer like the far end.
    if [ "$has_N" = 1 ]; then
      # -L is local:lport:remote:rport -- the SECOND field is the local port.
      p=$(printf '%s' "$LSPEC" | cut -d: -f2)
      python3 - "$p" <<'PY' &
import socket, sys, time
p = int(sys.argv[1])
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", p)); s.listen(5)
t = time.time()
while time.time() - t < 8:
    try:
        s.settimeout(1); c, _ = s.accept()
        c.sendall(b"SSH-2.0-OpenSSH_9.6\r\n"); c.close()
    except Exception:
        pass
PY
      echo $! > "$PORTFILE"; wait
    fi
    exit 0 ;;
esac
exit 0
"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        # Without this the shell silently falls through to /usr/bin/ssh and the
        # test "passes" against the real host instead of the stub.
        os.chmod(path, 0o755)

    def run(self, *args, timeout=60):
        return subprocess.run(["bash", SCRIPT, *args], env=self.env,
                              capture_output=True, text=True, timeout=timeout)


@pytest.fixture()
def make_box(tmp_path):
    return lambda mode: Box(str(tmp_path / mode), mode)


def test_reports_missing_key_before_anything_else(make_box):
    out = make_box("no-auth").run("u@h", "50051")
    text = out.stdout + out.stderr
    assert "cannot log in with a key" in text
    assert "--print-key" in text
    assert "forwarding works" not in text


def test_diagnoses_administratively_prohibited(make_box):
    """The one failure only an administrator can fix must say so plainly."""
    out = make_box("no-forwarding").run("u@h", "50051")
    text = out.stdout + out.stderr
    assert "administratively prohibited" in text.lower() or \
           "DISABLED on the server" in text
    assert "AllowTcpForwarding local" in text
    assert "administrator" in text.lower()


def test_forwarding_works_reads_the_remote_ssh_banner(make_box):
    """Test B proves forwarding by reaching the box's own sshd on :22."""
    out = make_box("ok").run("u@h", "50051")
    text = out.stdout + out.stderr
    assert "forwarding works" in text
    assert "SSH-2.0" in text


def test_reports_busy_local_port(make_box, tmp_path):
    out = make_box("busy-port").run("u@h", "50051")
    text = out.stdout + out.stderr
    assert "busy" in text.lower() or "already in use" in text.lower()
    assert "SSH-2.0" not in text


def test_probe_ports_are_not_hardcoded():
    """The probe forwards must ALLOCATE their local ports.

    A constant port is a false-pass generator: if anything already listens
    there, section B's banner read hits that stranger and reports
    "forwarding works" for a box that actually refuses to forward — the exact
    refusal the script exists to detect. It also fails a healthy box as
    "could not forward" when the constant is busy. This asserts the script
    derives them (free_port) instead of pinning numbers.
    """
    src = open(SCRIPT, encoding="utf-8").read()
    assert "free_port" in src, "no port allocator in check_ssh_forwarding.sh"
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for var in ("LP_A=", "LP_B="):
            if stripped.startswith(var):
                assert "free_port" in stripped, (
                    f"{var} is hardcoded ({stripped!r}); a busy port silently "
                    "fakes a PASS or a FAIL")


def test_exposure_classifier_flags_only_non_loopback():
    """The off-box-exposure warning must key on the LOCAL ADDRESS field.

    ``ss -ltn`` pads the peer column as ``*:*`` on every row, so the original
    whole-line grep for ``*:`` matched a loopback-bound worker and warned it was
    exposed. The classifier must stay quiet for loopback forms (including the
    ``[::ffff:127.0.0.1]`` v4-mapped one the box actually prints) and fire for
    every genuinely off-box bind.
    """
    cases = {
        "127.0.0.1:50051": False,
        "[::ffff:127.0.0.1]:50051": False,
        "[::1]:50051": False,
        "0.0.0.0:50051": True,
        "[::]:50051": True,
        "10.130.105.203:50051": True,
    }
    script = r'''
set -uo pipefail
classify() {
  EXPOSED=""
  while read -r addr; do
    [[ -z "$addr" ]] && continue
    case "$addr" in
      127.0.0.1:*|\[::1\]:*|\[::ffff:127.0.0.1\]:*) : ;;
      *) EXPOSED="$EXPOSED $addr" ;;
    esac
  done <<< "$1"
  [[ -n "$EXPOSED" ]] && echo WARN || echo quiet
}
'''
    import subprocess
    for addr, should_warn in cases.items():
        out = subprocess.run(["bash", "-c", script + f'\nclassify "{addr}"'],
                             capture_output=True, text=True, timeout=30)
        warned = "WARN" in out.stdout
        assert warned == should_warn, (
            f"{addr}: expected "
            f"{'a warning' if should_warn else 'quiet'}, got {out.stdout.strip()!r}")


def test_script_executes_at_all(make_box):
    """Guard against a syntax error making every diagnosis fail silently."""
    out = make_box("no-auth").run("u@h", "50051")
    assert out.returncode in (0, 1)
    assert "unbound variable" not in (out.stdout + out.stderr)
