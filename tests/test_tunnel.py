"""Tests for the SSH-only transport (remote/tunnel.py).

The real thing would open an SSH connection to the GPU box, which this agent
must never do. So these tests exercise the parts that decide *what* happens:
the exact ssh command line, the mode resolution, and the reuse/failure logic --
with subprocess and the port probe stubbed.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remote import settings, tunnel  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point the settings file at a temp path and clear tunnel env."""
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "remote.json"))
    monkeypatch.delenv("LT_SSH_TUNNEL", raising=False)
    monkeypatch.delenv("LT_SSH_PORT", raising=False)
    monkeypatch.delenv("LT_REMOTE_WORKER", raising=False)
    monkeypatch.delenv("LT_TUNNEL_PORT", raising=False)
    monkeypatch.delenv("LT_WORKER_MODE", raising=False)
    # set_worker_mode("local") suspends the tunnel by design; that flag is
    # module state, so a test that flips the mode would otherwise disable the
    # forward for every test after it.
    tunnel.resume()
    yield
    tunnel.resume()


# ─── the command line (the part that must be right) ───────────────────────
def test_command_has_loopback_forward_and_no_prompt():
    cmd = tunnel.build_command("ecpkn_bac", "10.130.105.203",
                              ssh_port=22, local_port=50051, remote_port=50051)
    assert cmd[0] == "ssh"
    assert "-N" in cmd
    assert "-L" in cmd
    i = cmd.index("-L")
    assert cmd[i + 1] == "127.0.0.1:50051:127.0.0.1:50051"
    # BatchMode: unattended, so a missing key must fail, never prompt.
    assert any("BatchMode=yes" in part for part in cmd)
    # ExitOnForwardFailure: a taken local port must fail, not hang connected.
    assert any("ExitOnForwardFailure=yes" in part for part in cmd)
    assert cmd[-1] == "ecpkn_bac@10.130.105.203"


def test_command_respects_custom_ssh_port():
    cmd = tunnel.build_command("u", "h", ssh_port=2222)
    assert cmd[cmd.index("-p") + 1] == "2222"


def test_command_without_username():
    cmd = tunnel.build_command("", "h")
    assert cmd[-1] == "h"


# ─── mode resolution ──────────────────────────────────────────────────────
def test_tunnel_on_by_default():
    assert settings.tunnel_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_tunnel_can_be_disabled_by_env(monkeypatch, value):
    monkeypatch.setenv("LT_SSH_TUNNEL", value)
    assert settings.tunnel_enabled() is False


def test_tunnel_round_trips_through_the_settings_file():
    settings.set_tunnel(False)
    assert settings.tunnel_enabled() is False
    settings.set_tunnel(True)
    assert settings.tunnel_enabled() is True


def test_effective_is_empty_without_a_target():
    """Tunnel mode must not invent 127.0.0.1:50051 for an unconfigured box."""
    address, source = settings.effective()
    assert address == ""
    assert source == settings.SOURCE_NONE


def test_effective_returns_tunnel_address_when_configured():
    settings.save_address("10.130.105.203:50051")
    address, source = settings.effective()
    assert address == "127.0.0.1:50051"
    assert source == settings.SOURCE_TUNNEL


def test_effective_returns_direct_address_when_tunnel_off():
    settings.save_address("10.130.105.203:50051")
    settings.set_tunnel(False)
    address, source = settings.effective()
    assert address == "10.130.105.203:50051"
    assert source == settings.SOURCE_UI


def test_effective_uses_env_target_in_tunnel_mode(monkeypatch):
    monkeypatch.setenv("LT_REMOTE_WORKER", "10.0.0.5:50051")
    address, source = settings.effective()
    assert address == "127.0.0.1:50051"
    assert source == settings.SOURCE_TUNNEL


# ─── compute mode (local / remote toggle) ─────────────────────────────────
def test_worker_mode_defaults_to_local():
    """Out of the box, GPU work stays on this machine."""
    assert settings.worker_mode() == settings.MODE_LOCAL
    assert settings.remote_active() is False


def test_worker_mode_round_trips_through_the_settings_file():
    settings.set_worker_mode("remote")
    assert settings.worker_mode() == settings.MODE_REMOTE
    assert settings.remote_active() is True
    settings.set_worker_mode("local")
    assert settings.worker_mode() == settings.MODE_LOCAL
    assert settings.remote_active() is False


def test_saved_worker_mode_beats_the_env_var(monkeypatch):
    """The UI toggle is the switch the user just flipped: it must win."""
    monkeypatch.setenv("LT_WORKER_MODE", "remote")
    settings.set_worker_mode("local")
    assert settings.worker_mode() == settings.MODE_LOCAL


def test_env_sets_the_mode_when_nothing_is_saved(monkeypatch):
    monkeypatch.setenv("LT_WORKER_MODE", "remote")
    assert settings.worker_mode() == settings.MODE_REMOTE


def test_cli_default_mode_is_the_last_fallback(monkeypatch):
    monkeypatch.delenv("LT_WORKER_MODE", raising=False)
    settings.set_default_mode("remote")
    try:
        assert settings.worker_mode() == settings.MODE_REMOTE
    finally:
        settings.set_default_mode("local")


def test_a_configured_address_does_not_turn_the_worker_on():
    """Having a box reachable must never move work off this machine."""
    settings.save_address("10.130.105.203:50051")
    assert settings.remote_active() is False


@pytest.mark.parametrize("bad", ["", "worker", "REMOTE!", "gpu"])
def test_invalid_worker_mode_is_rejected(bad):
    with pytest.raises(ValueError):
        settings.set_worker_mode(bad)


# ─── ensure(): reuse, start, failure ──────────────────────────────────────
class _FakeProc:
    pid = 4242
    returncode = None

    def __init__(self, rc=None):
        self._rc = rc
        self.returncode = rc

    def poll(self):
        return self._rc

    def terminate(self):
        self._rc = 0
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._rc = -9


def test_ensure_reuses_an_already_listening_port(monkeypatch):
    # A port already forwarded (e.g. a manual `ssh -L` you opened) is reused
    # rather than fighting over the local port.
    settings._write_file({"username": "u", "host": "h", "address": "h:50051"})
    monkeypatch.setattr(tunnel, "_port_open", lambda *a, **k: True)
    monkeypatch.setattr(tunnel.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("must not start a second ssh"))
    assert tunnel.ensure() is True


def test_ensure_wont_guess_without_a_configured_box(monkeypatch):
    """No host configured -> no tunnel. Do not silently forward to nowhere."""
    monkeypatch.setattr(tunnel.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("must not start ssh"))
    assert tunnel.ensure() is False


def test_ensure_starts_ssh_and_waits_for_the_port(monkeypatch):
    settings.save_credentials  # noqa: B018 -- settings file has no host yet
    settings._write_file({"username": "u", "host": "h", "address": "h:50051"})
    calls = {"n": 0}

    def port_open(*a, **k):
        calls["n"] += 1
        return calls["n"] > 1          # closed first, open after start

    started = {}

    def fake_popen(cmd, **kwargs):
        started["cmd"] = cmd
        return _FakeProc(rc=None)

    monkeypatch.setattr(tunnel, "_port_open", port_open)
    monkeypatch.setattr(tunnel.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tunnel.time, "sleep", lambda s: None)

    assert tunnel.ensure(timeout=2) is True
    assert started["cmd"][0] == "ssh"
    assert "u@h" in started["cmd"]


def test_ensure_reports_failure_when_ssh_exits(monkeypatch):
    settings._write_file({"username": "u", "host": "h", "address": "h:50051"})
    monkeypatch.setattr(tunnel, "_port_open", lambda *a, **k: False)
    monkeypatch.setattr(tunnel.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(rc=255))
    assert tunnel.ensure(timeout=1) is False


def test_stop_terminates_the_owned_child(monkeypatch):
    proc = _FakeProc(rc=None)
    monkeypatch.setattr(tunnel, "_proc", proc)
    tunnel.stop()
    assert tunnel._proc is None
    assert proc.poll() == 0


# ─── suspend / resume (the Disconnect button) ─────────────────────────────
def test_suspend_keeps_the_tunnel_down(monkeypatch):
    """Disconnect must outlive the supervisor, or the forward reopens at once."""
    settings._write_file({"username": "u", "host": "h", "address": "h:50051"})
    monkeypatch.setattr(tunnel, "_port_open", lambda *a, **k: False)
    monkeypatch.setattr(tunnel.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("suspended: must not start ssh"))
    tunnel.resume()
    tunnel.suspend()
    try:
        assert tunnel.is_suspended() is True
        assert tunnel.ensure() is False
        assert tunnel.status()["suspended"] is True
    finally:
        tunnel.resume()
    assert tunnel.is_suspended() is False
