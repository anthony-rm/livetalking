"""Secure storage for the remote GPU host's SSH credentials.

What goes where
---------------
* **password**  -> the OS keychain (``keyring``, service
  ``livetalking.remote-worker``). On macOS that is Keychain Access, on Linux
  SecretService/KWallet, on Windows the Credential Manager. It is NEVER written
  to a file in the repo, to .env, or to any JSON we own.
* **username / host / worker address** -> ``data/remote_worker.json``. These are
  not secrets (they also live in ~/.ssh/config or a LAN scan), so they sit in the
  same small settings file. ``data/`` is gitignored.

Address resolution order (unchanged):

    1. address saved from the admin console  (data/remote_worker.json)
    2. --remote_worker CLI flag / LT_REMOTE_WORKER env var
    3. none -> remote features disabled

Threat model, stated honestly
-----------------------------
This protects the password **at rest** and keeps it out of git, logs, API
responses, and shell history. It does NOT protect against a process running as
your user that can talk to the keychain, and it does NOT encrypt traffic --
use SSH or a tunnel for that. The admin UI is unauthenticated (as is the rest of
this app), so anyone who can reach the port can ask the server to use the stored
credentials. Do not expose this server to an untrusted network.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

from utils.logger import logger

SETTINGS_PATH = os.path.abspath("./data/remote_worker.json")
KEYRING_SERVICE = "livetalking.remote-worker"

SOURCE_UI = "admin-ui"
SOURCE_ENV = "cli/env"
SOURCE_TUNNEL = "ssh-tunnel"
SOURCE_NONE = "unset"

# Compute mode: where GPU work runs. ``local`` (the default) keeps every task
# on this machine; ``remote`` sends inference, avatar generation and enhance to
# the worker. An address alone does NOT enable the worker -- having a box
# configured must never silently move work off this machine.
MODE_LOCAL = "local"
MODE_REMOTE = "remote"
_MODES = (MODE_LOCAL, MODE_REMOTE)

# Fallback set from --worker_mode at startup. Only used when neither the env
# var nor the saved value decides, so the UI toggle always wins over a stale
# command line.
_default_mode = MODE_LOCAL


# ─── settings file (non-secret parts) ─────────────────────────────────────
def _read_file() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001
        logger.warning("[remote] could not read %s", SETTINGS_PATH, exc_info=True)
        return {}


def _write_file(data: dict) -> None:
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, SETTINGS_PATH)   # atomic: a crash never leaves a half file


# ─── validation ───────────────────────────────────────────────────────────
def normalize(address: Optional[str]) -> str:
    """Trim and validate a host:port string. Returns '' for empty input."""
    addr = (address or "").strip()
    if not addr:
        return ""
    if addr.count(":") != 1:
        raise ValueError(
            f"expected host:port (e.g. 192.168.1.42:50051), got {addr!r}")
    host, port = addr.split(":")
    if not host:
        raise ValueError(f"missing host in {addr!r}")
    try:
        port_i = int(port)
    except ValueError:
        raise ValueError(f"port is not a number in {addr!r}") from None
    if not (1 <= port_i <= 65535):
        raise ValueError(f"port out of range in {addr!r}")
    return f"{host}:{port_i}"


# ─── compute mode (local vs remote worker) ────────────────────────────────
# Resolution order, mirroring the address: a value saved from the UI wins (it
# is the switch the user just flipped), then LT_WORKER_MODE in the environment,
# then --worker_mode on the command line.
def worker_mode() -> str:
    """'local' or 'remote' — where GPU work should run."""
    data = _read_file()
    saved = str(data.get("worker_mode") or "").strip().lower()
    if saved in _MODES:
        return saved
    env = os.environ.get("LT_WORKER_MODE", "").strip().lower()
    if env in _MODES:
        return env
    return _default_mode if _default_mode in _MODES else MODE_LOCAL


def set_worker_mode(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode not in _MODES:
        raise ValueError(f"worker mode must be one of {list(_MODES)}, got {mode!r}")
    data = _read_file()
    data["worker_mode"] = mode
    _write_file(data)
    logger.info("[remote] worker mode set to %s", mode)
    if mode == MODE_LOCAL:
        # Leaving the worker on also leaves its SSH forward up and its GPU
        # memory held. Drop the forward (the supervisor respects the suspend)
        # so 'local' means what it says.
        try:
            from remote import tunnel
            tunnel.suspend()
        except Exception:  # noqa: BLE001
            logger.debug("[remote] tunnel suspend failed", exc_info=True)
    elif mode == MODE_REMOTE:
        try:
            from remote import tunnel
            tunnel.resume()
        except Exception:  # noqa: BLE001
            logger.debug("[remote] tunnel resume failed", exc_info=True)
    return mode


def set_default_mode(mode: str) -> None:
    """Record the --worker_mode value as the fallback for worker_mode()."""
    global _default_mode
    mode = (mode or "").strip().lower()
    if mode in _MODES:
        _default_mode = mode


def remote_active() -> bool:
    """True when work must be dispatched to the worker (the toggle says remote).

    Deliberately only the toggle: a user who picks "remote" and has not
    connected a box is misconfigured, and the dispatchers fail with a clear
    "no worker address" error instead of silently running a multi-hour job on
    this machine. Killing sessions is slow enough already.
    """
    return worker_mode() == MODE_REMOTE


# ─── SSH-only transport (tunnel) ──────────────────────────────────────────
# Default ON: the worker protocol is plaintext and unauthenticated, so the
# supported way to reach it is an SSH local-forward. Turning this off means the
# host dials the worker address directly (trusted LAN only).
def tunnel_enabled() -> bool:
    """True when the worker should be reached through an SSH tunnel."""
    env = os.environ.get("LT_SSH_TUNNEL")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off")
    data = _read_file()
    if "ssh_tunnel" in data:
        return bool(data["ssh_tunnel"])
    return True


def set_tunnel(enabled: bool) -> None:
    data = _read_file()
    data["ssh_tunnel"] = bool(enabled)
    _write_file(data)
    logger.info("[remote] SSH tunnel mode %s", "enabled" if enabled else "disabled")


def ssh_port() -> str:
    """SSH port for the tunnel (default 22). Stored in the settings file."""
    env = os.environ.get("LT_SSH_PORT")
    if env:
        return env
    data = _read_file()
    port = data.get("ssh_port")
    return str(port) if port else "22"


def set_ssh_port(port: int) -> None:
    data = _read_file()
    data["ssh_port"] = int(port)
    _write_file(data)


def worker_port() -> int:
    """Port the worker listens on (loopback) on the box. Default 50051."""
    data = _read_file()
    port = data.get("worker_port")
    return int(port) if port else int(os.environ.get("LT_REMOTE_PORT", "50051"))


# ─── effective worker address ─────────────────────────────────────────────
def saved_address() -> str:
    """Worker address from settings.

    ``address`` is the explicit host:port. Files written by ``save_credentials``
    have only ``username``/``host`` (the admin console's normal path), so fall
    back to ``host:worker_port()``. Without this, ``effective()`` reported
    "unset" for a fully configured box and the avatar plugin refused to run.
    """
    try:
        addr = normalize(_read_file().get("address") or "")
    except ValueError:
        logger.warning("[remote] ignoring malformed saved address")
        addr = ""
    if addr:
        return addr
    host = (_read_file().get("host") or "").strip()
    if not host:
        return ""
    try:
        return normalize(f"{host}:{worker_port()}")
    except ValueError:
        logger.warning("[remote] ignoring malformed saved host")
        return ""


def env_address() -> str:
    try:
        return normalize(os.environ.get("LT_REMOTE_WORKER", ""))
    except ValueError:
        logger.warning("[remote] ignoring malformed LT_REMOTE_WORKER")
        return ""


def effective() -> Tuple[str, str]:
    """Return (address, source). Address is '' when unconfigured.

    In SSH-only mode (the default) the address is the LOCAL end of the tunnel,
    because that is what the client must dial. The saved/CLI address still
    identifies WHICH worker, and is kept in the file for the tunnel to target.
    """
    if tunnel_enabled():
        from remote import tunnel
        # No target configured at all -> no address. Without this check the
        # tunnel mode would hand out 127.0.0.1:50051 for a box the user never
        # set, and the UI would show "configured" for a worker that cannot exist.
        if not saved_address() and not env_address():
            return "", SOURCE_NONE
        return tunnel.local_address(tunnel.DEFAULT_LOCAL_PORT), SOURCE_TUNNEL
    saved = saved_address()
    if saved:
        return saved, SOURCE_UI
    env = env_address()
    if env:
        return env, SOURCE_ENV
    return "", SOURCE_NONE


def effective_address() -> str:
    return effective()[0]


def save_address(address: str) -> str:
    """Persist the worker address. Returns the normalised value."""
    addr = normalize(address)
    data = _read_file()
    data["address"] = addr
    _write_file(data)
    logger.info("[remote] saved worker address %s (admin ui)", addr)
    return addr


def clear_saved() -> None:
    """Remove the address AND the credentials (they belong to that address).

    Pinned SSH host keys are deliberately kept: re-pinning after every address
    change would train you to click through the warning that matters.
    """
    clear_credentials()
    data = _read_file()
    data.pop("address", None)
    if any(k in data for k in ("username", "host")):
        _write_file(data)
    elif data.get("host_keys"):
        _write_file(data)
    else:
        try:
            os.remove(SETTINGS_PATH)
        except FileNotFoundError:
            pass
    logger.info("[remote] cleared saved worker address and credentials")


# ─── credentials ──────────────────────────────────────────────────────────
def _keyring_set(username: str, password: str) -> None:
    try:
        import keyring
        keyring.set_password(KEYRING_SERVICE, username, password)
    except ImportError as exc:
        raise RuntimeError(
            "the 'keyring' package is required to store the password securely "
            "(pip install keyring)") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not write to the OS keychain: {exc}") from exc


def save_credentials(username: str, host: str, password: str) -> None:
    """Store username/host in the settings file, password in the OS keychain."""
    username = (username or "").strip()
    host = (host or "").strip()
    password = password or ""
    if not username:
        raise ValueError("username is required")
    if not host:
        raise ValueError("host is required")
    if not password:
        raise ValueError("password is required")

    _keyring_set(username, password)

    data = _read_file()
    data["username"] = username
    data["host"] = host
    _write_file(data)
    logger.info("[remote] saved credentials for %s@%s (password in OS keychain)",
                username, host)


def get_credentials() -> Optional[dict]:
    """Return {username, host, password} or None. Never logs the password."""
    data = _read_file()
    username = (data.get("username") or "").strip()
    host = (data.get("host") or "").strip()
    if not username or not host:
        return None
    password = ""
    try:
        import keyring
        password = keyring.get_password(KEYRING_SERVICE, username) or ""
    except Exception:  # noqa: BLE001
        logger.warning("[remote] could not read keychain entry", exc_info=True)
    if not password:
        return None
    return {"username": username, "host": host, "password": password}


def credentials_summary() -> dict:
    """Safe-to-serialise description: never includes the password itself."""
    data = _read_file()
    username = (data.get("username") or "").strip()
    host = (data.get("host") or "").strip()
    has_password = False
    if username:
        try:
            import keyring
            has_password = bool(keyring.get_password(KEYRING_SERVICE, username))
        except Exception:  # noqa: BLE001
            has_password = False
    return {
        "username": username,
        "host": host,
        "has_password": has_password,
        "complete": bool(username and host and has_password),
    }


def identity() -> Tuple[str, str]:
    """(username, host) from the settings file. Not secret; safe to log.

    The deploy tool needs the same SSH identity the admin console manages, but
    it must not reach into ``_read_file`` to get it.
    """
    data = _read_file()
    return ((data.get("username") or "").strip(),
            (data.get("host") or "").strip())


def clear_credentials() -> None:
    data = _read_file()
    username = (data.get("username") or "").strip()
    if username:
        try:
            import keyring
            keyring.delete_password(KEYRING_SERVICE, username)
        except Exception:  # noqa: BLE001
            # A missing entry raises; that is fine (idempotent clear).
            logger.debug("[remote] no keychain entry to delete for %s", username)
    for key in ("username", "host"):
        data.pop(key, None)
    if data:
        _write_file(data)


# ─── pinned SSH host keys ─────────────────────────────────────────────────
# Trust-on-first-use: the first successful SSH test records the fingerprint, and
# every later test compares against it so a changed host key (possible MITM, or
# a reinstalled PC) is a loud failure rather than a silent reconnect.
def pinned_host_keys() -> dict:
    keys = _read_file().get("host_keys")
    return keys if isinstance(keys, dict) else {}


def pin_host_key(host: str, fingerprint: str) -> None:
    """Record a host key. No-op when the same fingerprint is already pinned.

    Re-pinning on every successful test rewrote the settings file and logged a
    line each time, which made the log look like the key kept changing.
    """
    host = (host or "").strip()
    data = _read_file()
    keys = data.get("host_keys")
    if not isinstance(keys, dict):
        keys = {}
    if keys.get(host) == fingerprint:
        return
    keys[host] = fingerprint
    data["host_keys"] = keys
    _write_file(data)
    logger.info("[remote] pinned SSH host key for %s (%s...)", host, fingerprint[:16])


def unpin_host_key(host: str) -> None:
    data = _read_file()
    keys = data.get("host_keys")
    if isinstance(keys, dict) and (host or "").strip() in keys:
        keys.pop((host or "").strip(), None)
        data["host_keys"] = keys
        _write_file(data)
