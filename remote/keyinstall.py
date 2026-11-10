"""Make this host's SSH key usable on the GPU box, so the tunnel can log in.

Why this exists
---------------
The worker binds loopback on the box and is only reachable through an
``ssh -N -L`` forward (see ``remote/tunnel.py``). That forward runs unattended
with ``BatchMode=yes``: it cannot answer a password prompt. A password is
therefore only good for logging IN *once* -- it is what lets the admin console
append one line to the box's ``authorized_keys``, after which the forward
authenticates with the key like any other ssh command.

This is the same idempotent, no-sudo operation as ``./deploy.sh --setup-key``,
just reachable from the "Connect" button. Nothing else on the box is touched:
the remote command appends the exact key line only when it is missing.

The password is only ever handed to paramiko in memory (never argv, never
``sshpass``), and the host key is compared against the already-pinned
fingerprint before anything is written -- a changed key is refused rather than
silently trusted.
"""
from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from typing import Optional

from utils.logger import logger

# Public keys to look for, in order of preference.
KEY_NAMES = ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub")

# Idempotent append. The markers let the caller tell "already authorized" from
# "just installed", which is the difference between a no-op and a change.
_INSTALL_CMD = (
    "umask 077; mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && "
    "if grep -qxF \"$LT_KEY\" ~/.ssh/authorized_keys; then echo __LT_ALREADY__; "
    "else printf '%s\\n' \"$LT_KEY\" >> ~/.ssh/authorized_keys; echo __LT_INSTALLED__; fi"
)


def public_key() -> str:
    """The first usable public key in ~/.ssh. Raises RuntimeError when absent."""
    for name in KEY_NAMES:
        path = os.path.expanduser(f"~/.ssh/{name}")
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read().strip()
        except OSError:
            continue
        if text:
            return text
    raise RuntimeError(
        "no SSH public key found in ~/.ssh (looked for "
        + ", ".join(KEY_NAMES)
        + "). Create one with: ssh-keygen -t ed25519 -C \"$(whoami)@$(hostname)\"")


def ensure_installed(username: str, host: str, password: str, *, port: int = 22,
                     expected_fingerprint: Optional[str] = None,
                     timeout: float = 15.0) -> dict:
    """Authorize this host's public key on the box. Never raises.

    Result keys: ok, already_present, fingerprint, message.
    """
    try:
        import paramiko
    except ImportError:
        return {"ok": False, "message": "paramiko is not installed (pip install paramiko)"}
    try:
        key = public_key()
    except RuntimeError as exc:
        return {"ok": False, "message": str(exc)}

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=host, port=port, username=username,
                       password=password, timeout=timeout, auth_timeout=timeout,
                       banner_timeout=timeout, allow_agent=False,
                       look_for_keys=False)
    except paramiko.AuthenticationException:
        return {"ok": False, "message": "authentication failed — wrong username or password"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"could not log in: {exc}"}

    try:
        transport = client.get_transport()
        if transport is None:
            return {"ok": False, "message": "SSH transport closed before the host key was read"}
        fp = hashlib.sha256(transport.get_remote_server_key().asbytes()).hexdigest()
        if expected_fingerprint and expected_fingerprint != fp:
            logger.error("[remote] host key for %s changed (%s -> %s); refusing",
                         host, expected_fingerprint[:16], fp[:16])
            return {"ok": False, "fingerprint": fp,
                    "message": "HOST KEY CHANGED — refusing to write anything. Expected "
                               f"{expected_fingerprint[:16]}..., got {fp[:16]}... "
                               "Possible man-in-the-middle; verify the box first."}

        _stdin, stdout, stderr = client.exec_command(
            f"LT_KEY={shlex.quote(key)} bash -c {shlex.quote(_INSTALL_CMD)}",
            timeout=timeout)
        try:
            out = stdout.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 -- channel closed early; markers absent -> handled below
            out = ""
        err = stderr.read().decode("utf-8", "replace").strip()
        already = "__LT_ALREADY__" in out
        installed = "__LT_INSTALLED__" in out
        if err:
            return {"ok": False, "fingerprint": fp, "message": f"key install failed: {err}"}
        if not (already or installed):
            return {"ok": False, "fingerprint": fp,
                    "message": "key install produced no confirmation — check that the box "
                               "has bash and a writable ~/.ssh"}
        logger.info("[remote] public key %s for %s@%s",
                    "already authorized" if already else "installed", username, host)
        return {"ok": True, "already_present": already, "fingerprint": fp,
                "message": ("already authorized" if already
                            else "public key appended to ~/.ssh/authorized_keys")}
    finally:
        client.close()


def verify_key_login(username: str, host: str, *, port: int = 22,
                     timeout: float = 15.0) -> dict:
    """Prove that a non-interactive, key-only login works -- what the tunnel needs.

    Deliberately uses the same options as ``remote/tunnel.build_command`` (so a
    success here predicts the forward working) and never asks for anything.
    """
    user = f"{username}@" if username else ""
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-p", str(port), f"{user}{host}", "true"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "message": f"could not run ssh: {exc}"}
    if proc.returncode == 0:
        return {"ok": True, "message": "key-based login works"}
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    first = detail[0] if detail else f"ssh exited {proc.returncode}"
    return {"ok": False, "message": first[:300]}
