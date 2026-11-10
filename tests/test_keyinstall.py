"""Tests for remote/keyinstall.py — the "Connect" button's key bootstrap.

The real function talks to an SSH server, which these tests must not do. A stub
paramiko stands in for the transport so the parts that can silently go wrong are
checked: the exact remote command (idempotent append, no sudo), the host-key
comparison, and the marker-based result.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remote import keyinstall  # noqa: E402


class _Stdin:
    def write(self, _data):
        pass

    def close(self):
        pass


class _Stream:
    def __init__(self, data: str = ""):
        self._data = data.encode()

    def read(self):
        return self._data


class _Transport:
    def __init__(self, secret: bytes = b"hostkey"):
        self._secret = secret

    def get_remote_server_key(self):
        return types.SimpleNamespace(asbytes=lambda: self._secret)


class _Client:
    """Records the command it was asked to run and replies with fixed output."""

    last: "_Client | None" = None
    stdout = "__LT_INSTALLED__\n"
    stderr = ""

    def __init__(self):
        _Client.last = self
        self.commands = []
        self.closed = False
        self.connect_kwargs = {}

    def set_missing_host_key_policy(self, _policy):
        pass

    def connect(self, **kwargs):
        self.connect_kwargs = kwargs

    def get_transport(self):
        return _Transport()

    def exec_command(self, command, timeout=None):
        self.commands.append(command)
        return _Stdin(), _Stream(_Client.stdout), _Stream(_Client.stderr)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_paramiko(monkeypatch, tmp_path):
    """Install a stub paramiko and a temp ~/.ssh with one public key."""
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test@host\n")
    monkeypatch.setenv("HOME", str(tmp_path))

    mod = types.SimpleNamespace(
        SSHClient=_Client,
        AutoAddPolicy=lambda: object(),
        AuthenticationException=type("AuthenticationException", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "paramiko", mod)
    _Client.last = None
    _Client.stdout = "__LT_INSTALLED__\n"
    _Client.stderr = ""
    return mod


def test_public_key_reads_the_first_available_key(fake_paramiko):
    assert keyinstall.public_key().startswith("ssh-ed25519 AAAA")


def test_public_key_raises_without_any_key(fake_paramiko, tmp_path):
    os.remove(tmp_path / ".ssh" / "id_ed25519.pub")
    with pytest.raises(RuntimeError, match="no SSH public key"):
        keyinstall.public_key()


def test_install_appends_without_sudo_and_is_idempotent(fake_paramiko):
    result = keyinstall.ensure_installed("u", "h", "pw")
    assert result["ok"] is True
    assert result["already_present"] is False
    cmd = _Client.last.commands[0]
    # Idempotent guard, not a blind append.
    assert "grep -qxF" in cmd
    assert ">> ~/.ssh/authorized_keys" in cmd
    # No privilege escalation: this must work for an unprivileged account.
    assert "sudo" not in cmd
    assert _Client.last.closed is True


def test_already_authorized_is_reported_as_such(fake_paramiko):
    _Client.stdout = "__LT_ALREADY__\n"
    result = keyinstall.ensure_installed("u", "h", "pw")
    assert result["ok"] is True
    assert result["already_present"] is True


def test_password_is_passed_in_memory_not_on_a_command_line(fake_paramiko):
    keyinstall.ensure_installed("u", "h", "Sup3rS3cret")
    assert _Client.last.connect_kwargs["password"] == "Sup3rS3cret"
    assert all("Sup3rS3cret" not in c for c in _Client.last.commands)


def test_changed_host_key_writes_nothing(fake_paramiko):
    result = keyinstall.ensure_installed("u", "h", "pw",
                                        expected_fingerprint="deadbeef")
    assert result["ok"] is False
    assert "HOST KEY CHANGED" in result["message"]
    assert _Client.last.commands == []


def test_missing_confirmation_is_a_failure(fake_paramiko):
    _Client.stdout = ""
    result = keyinstall.ensure_installed("u", "h", "pw")
    assert result["ok"] is False
    assert "no confirmation" in result["message"]


def test_auth_failure_is_reported_plainly(fake_paramiko, monkeypatch):
    def boom(self, **kwargs):
        raise fake_paramiko.AuthenticationException()

    monkeypatch.setattr(_Client, "connect", boom)
    result = keyinstall.ensure_installed("u", "h", "wrong")
    assert result["ok"] is False
    assert "authentication failed" in result["message"]


def test_verify_key_login_never_prompts(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(keyinstall.subprocess, "run", fake_run)
    assert keyinstall.verify_key_login("u", "h", port=2222)["ok"] is True
    assert "BatchMode=yes" in seen["cmd"]
    assert seen["cmd"][-2] == "u@h"
    assert seen["cmd"][-1] == "true"
    assert seen["cmd"][seen["cmd"].index("-p") + 1] == "2222"


def test_verify_key_login_surfaces_the_first_error_line(monkeypatch):
    def fake_run(cmd, **kwargs):
        return types.SimpleNamespace(
            returncode=255, stdout="",
            stderr="Permission denied (publickey).\nmore noise")

    monkeypatch.setattr(keyinstall.subprocess, "run", fake_run)
    result = keyinstall.verify_key_login("u", "h")
    assert result["ok"] is False
    assert result["message"] == "Permission denied (publickey)."
