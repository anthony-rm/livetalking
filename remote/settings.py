"""Secure storage for the Windows-PC connection credentials.

What goes where
---------------
* **password**  -> the OS keychain (``keyring``, service
  ``livetalking.remote-worker``). On macOS that is Keychain Access, on Windows
  the Credential Manager, on Linux SecretService/KWallet. It is NEVER written to
  a file in the repo, to .env, or to any JSON we own.
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
SOURCE_NONE = "unset"


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


# ─── effective worker address ─────────────────────────────────────────────
def saved_address() -> str:
    try:
        return normalize(_read_file().get("address") or "")
    except ValueError:
        logger.warning("[remote] ignoring malformed saved address")
        return ""


def env_address() -> str:
    try:
        return normalize(os.environ.get("LT_REMOTE_WORKER", ""))
    except ValueError:
        logger.warning("[remote] ignoring malformed LT_REMOTE_WORKER")
        return ""


def effective() -> Tuple[str, str]:
    """Return (address, source). Address is '' when unconfigured."""
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
    data = _read_file()
    keys = data.get("host_keys")
    if not isinstance(keys, dict):
        keys = {}
    keys[(host or "").strip()] = fingerprint
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
