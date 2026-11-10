"""SSH-only transport: keep the worker's gRPC port on the loopback interface.

Why this exists
---------------
The worker protocol is plaintext and unauthenticated. Binding it to
``0.0.0.0:50051`` means anything that can reach the port gets the GPU. Opening a
firewall hole for it is worse. So the default is now:

    worker listens on 127.0.0.1:50051   (loopback only, nothing exposed)
    host opens      ssh -N -L 50051:127.0.0.1:50051 user@box
    host talks to   127.0.0.1:50051     (i.e. through the encrypted channel)

Authentication and encryption are then SSH's job, not the RPC layer's, and the
only way in is the SSH account. The tunnel is a local port, so no firewall rule
is needed on either side.

Everything else in the app keeps using ``remote.settings.effective_address()``,
which returns the tunnel's local address when tunnel mode is on. That keeps the
avatar plugin, the enhance dispatcher, the deploy verifier and the admin console
agreeing on one address.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

from utils.logger import logger

LOG_PATH = os.path.abspath("./data/tunnel.log")
DEFAULT_LOCAL_PORT = int(os.environ.get("LT_TUNNEL_PORT", "50051"))
DEFAULT_REMOTE_PORT = int(os.environ.get("LT_REMOTE_PORT", "50051"))

_proc: Optional[subprocess.Popen] = None
_lock = threading.Lock()
# Set by "Disconnect" in the admin console. Without it the 10s supervisor would
# re-open a forward the user just closed, which is worse than no button at all.
_suspended = False


def suspend() -> None:
    """Stop forwarding and keep it stopped until resume()."""
    global _suspended
    _suspended = True
    stop()
    logger.info("[tunnel] suspended (disconnected from the admin console)")


def resume() -> None:
    global _suspended
    if _suspended:
        logger.info("[tunnel] resumed")
    _suspended = False


def is_suspended() -> bool:
    return _suspended



def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    """True when something is accepting connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def is_enabled() -> bool:
    from remote import settings
    return settings.tunnel_enabled()


def local_address(port: Optional[int] = None) -> str:
    port = port or DEFAULT_LOCAL_PORT
    return f"127.0.0.1:{port}"


def build_command(username: str, host: str, *, ssh_port: int = 22,
                  local_port: int = DEFAULT_LOCAL_PORT,
                  remote_port: int = DEFAULT_REMOTE_PORT) -> list:
    """The ssh invocation that forwards the worker port.

    Flags that matter:
      * ``-N``            no remote command; forward only
      * ``ExitOnForwardFailure``  fail fast if the local port is taken, instead
                          of sitting there connected-but-useless
      * ``ServerAlive*``  notice a dead link and exit, so the supervisor can
                          restart it, rather than hanging until TCP times out
      * ``BatchMode=yes`` never prompt: this runs unattended, so a missing key
                          must be an error, not a password prompt
    """
    user = f"{username}@" if username else ""
    return [
        "ssh", "-N",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        "-p", str(ssh_port),
        f"{user}{host}",
    ]


def status() -> dict:
    """Report without starting anything (used by the admin API)."""
    return {
        "enabled": is_enabled(),
        "suspended": _suspended,
        "running": _proc is not None and _proc.poll() is None,
        "listening": _port_open(DEFAULT_LOCAL_PORT),
        "address": local_address(),
        "pid": _proc.pid if _proc is not None and _proc.poll() is None else None,
    }


def ensure(*, timeout: float = 20.0, restart: bool = True) -> bool:
    """Make sure a tunnel to the worker is up. Returns True when usable.

    Safe to call repeatedly: a live child is kept, an already-listening local
    port (e.g. a tunnel you started by hand) is reused, and a dead child is
    replaced. Never raises -- callers log and continue, because the failure is
    an unreachable worker, not a crash.
    """
    global _proc
    if not is_enabled():
        return True                       # not tunnelling: direct/LAN mode
    if _suspended:
        return False                      # user pressed Disconnect

    from remote import settings
    username, host = settings.identity()
    if not host:
        logger.error("[tunnel] tunnel mode is on but no SSH host is configured "
                     "(admin console -> Remote Worker)")
        return False

    with _lock:
        if _port_open(DEFAULT_LOCAL_PORT):
            if _proc is None or _proc.poll() is not None:
                logger.info("[tunnel] %s already listening; reusing it",
                            local_address())
            return True

        if _proc is not None and _proc.poll() is None and not restart:
            return False

        cmd = build_command(username, host,
                            ssh_port=int(settings.ssh_port()),
                            local_port=DEFAULT_LOCAL_PORT)
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log = open(LOG_PATH, "ab")
        try:
            _proc = subprocess.Popen(cmd, stdout=log, stderr=log,
                                     stdin=subprocess.DEVNULL)
        except OSError as exc:
            logger.error("[tunnel] could not start ssh: %s", exc)
            return False
        logger.info("[tunnel] ssh -L 127.0.0.1:%d:127.0.0.1:%d -> %s (pid %d)",
                    DEFAULT_LOCAL_PORT, DEFAULT_REMOTE_PORT, host, _proc.pid)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _proc.poll() is not None:
            logger.error("[tunnel] ssh exited with %s; see %s",
                         _proc.returncode, LOG_PATH)
            return False
        if _port_open(DEFAULT_LOCAL_PORT):
            logger.info("[tunnel] forwarding %s -> box:127.0.0.1:%d",
                        local_address(), DEFAULT_REMOTE_PORT)
            return True
        time.sleep(0.25)
    logger.error("[tunnel] local port %d never opened; see %s",
                 DEFAULT_LOCAL_PORT, LOG_PATH)
    return False


def stop() -> None:
    """Close the tunnel this process owns. Leaves a hand-started one alone."""
    global _proc
    with _lock:
        if _proc is None or _proc.poll() is not None:
            _proc = None
            return
        pid = _proc.pid
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
        logger.info("[tunnel] closed (pid %d)", pid)
        _proc = None


_ensure_lock = threading.Lock()
_supervisor_started = False


def start_background_supervisor(interval: float = 10.0) -> None:
    """Keep the tunnel alive for the lifetime of the server process.

    A dropped link is normal (laptop sleep, Wi-Fi change, sshd restart); without
    this the server would stay "up" while every session failed. The supervisor
    is a daemon thread, so it never delays shutdown. Idempotent: startup and the
    admin console's mode toggle both call it.
    """
    global _supervisor_started
    if not is_enabled():
        return
    with _ensure_lock:
        if _supervisor_started:
            return
        _supervisor_started = True
    if ensure():
        logger.info("[tunnel] ready at %s", local_address())
    else:
        logger.warning("[tunnel] not up yet; will keep retrying every %gs",
                       interval)

    def _loop():
        while True:
            time.sleep(interval)
            try:
                ensure()
            except Exception:  # noqa: BLE001
                logger.warning("[tunnel] supervisor pass failed", exc_info=True)

    threading.Thread(target=_loop, name="ssh-tunnel", daemon=True).start()


def main(argv=None) -> int:
    """CLI: python -m remote.tunnel [start|stop|status]"""
    argv = list(sys.argv[1:] if argv is None else argv)
    action = argv[0] if argv else "status"
    if action == "start":
        ok = ensure()
        print("tunnel up at", local_address()) if ok else print("tunnel FAILED", file=sys.stderr)
        return 0 if ok else 1
    if action == "stop":
        stop()
        return 0
    info = status()
    for key, value in info.items():
        print(f"{key:10} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
