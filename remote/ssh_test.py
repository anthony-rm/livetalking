"""Verify Windows-PC SSH credentials on behalf of the admin console.

Used by the "Test SSH" button. paramiko authenticates over SSH and runs one
trivial command to fetch the remote hostname -- that proves the username,
password, host, and SSH server are all usable together, which a TCP probe
cannot.

Security notes
--------------
* The password is passed to paramiko in memory only. It is never put on a
  command line (no `sshpass -p`), so it cannot leak via `ps` or shell history.
* The remote command is a fixed string; nothing from the request is
  interpolated into it.
* We do NOT auto-accept host keys silently on every call: the caller passes the
  already-known fingerprint and we compare. ``trust_new`` is an explicit,
  user-visible opt-in for the first connection only.
"""
from __future__ import annotations

import socket
from typing import Optional, Tuple

from utils.logger import logger

DEFAULT_SSH_PORT = 22
_PROBE_CMD = "hostname"     # fixed; never built from request data


def _fingerprint(key) -> str:
    import hashlib
    return hashlib.sha256(key.asbytes()).hexdigest()


def test_ssh(username: str, host: str, password: str, *, port: int = DEFAULT_SSH_PORT,
             expected_fingerprint: Optional[str] = None,
             trust_new: bool = False, timeout: float = 10.0) -> dict:
    """Try to log in over SSH. Returns a result dict, never raises.

    Result keys: ok, host, port, hostname (remote), fingerprint, message.
    """
    try:
        import paramiko
    except ImportError:
        return {"ok": False, "message":
                "paramiko is not installed (pip install paramiko)"}

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())

    if trust_new:
        # First-contact convenience: accept and report, but the caller is told.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=host, port=port, username=username, password=password,
            timeout=timeout, auth_timeout=timeout,
            banner_timeout=timeout, allow_agent=False, look_for_keys=False)
    except paramiko.AuthenticationException:
        return {"ok": False, "host": host, "port": port,
                "message": "authentication failed — wrong username or password"}
    except paramiko.SSHException as exc:
        msg = str(exc)
        if "not found in known_hosts" in msg or "Host key" in msg:
            return {"ok": False, "host": host, "port": port, "needs_trust": True,
                    "message": "host key not trusted yet — confirm the fingerprint "
                               "to pin it"}
        return {"ok": False, "host": host, "port": port,
                "message": f"SSH error: {msg}"}
    except socket.timeout:
        return {"ok": False, "host": host, "port": port,
                "message": f"timed out after {timeout:g}s — host unreachable or "
                           f"port {port} closed"}
    except (socket.error, OSError) as exc:
        return {"ok": False, "host": host, "port": port,
                "message": f"connection failed: {exc}"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("[remote] ssh test unexpected error", exc_info=True)
        return {"ok": False, "host": host, "port": port,
                "message": f"{type(exc).__name__}: {exc}"}

    try:
        transport = client.get_transport()
        if transport is None:
            return {"ok": False, "host": host, "port": port,
                    "message": "SSH transport closed unexpectedly"}
        key = transport.get_remote_server_key()
        fp = _fingerprint(key)

        if expected_fingerprint and expected_fingerprint != fp:
            return {"ok": False, "host": host, "port": port, "fingerprint": fp,
                    "message": "HOST KEY CHANGED — refusing. Expected "
                               f"{expected_fingerprint[:16]}..., got {fp[:16]}... "
                               "Possible man-in-the-middle; verify the PC."}

        stdin, stdout, stderr = client.exec_command(_PROBE_CMD, timeout=timeout)
        remote_host = stdout.read().decode("utf-8", "replace").strip()
        return {"ok": True, "host": host, "port": port,
                "hostname": remote_host, "fingerprint": fp,
                "new_host_key": bool(trust_new and not expected_fingerprint),
                "message": f"connected to {remote_host or host}"}
    finally:
        client.close()


def seeded_known_hosts() -> dict:
    """{host: fingerprint} pinned by a previous successful test."""
    from remote import settings
    return settings.pinned_host_keys()
