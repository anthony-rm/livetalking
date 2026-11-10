"""Tests for the local/remote compute toggle (remote/settings.py + the API).

The toggle is the one switch that decides whether avatar inference, avatar
generation and video enhance run here or on the GPU box. These tests pin the
contract that matters:

* default is LOCAL, even with a worker address configured and reachable;
* the persisted UI value wins over env / CLI;
* every dispatcher asks the same question (``remote_active()``);
* POST /api/remote/mode persists the choice and reports a stale resident model.

No network, no GPU: the settings file is redirected to a temp path and the
tunnel is stubbed at the module boundary.
"""
from __future__ import annotations

import asyncio
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remote import settings  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "remote.json"))
    for var in ("LT_WORKER_MODE", "LT_SSH_TUNNEL", "LT_REMOTE_WORKER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(settings, "_default_mode", settings.MODE_LOCAL)
    # set_worker_mode touches the tunnel; keep it out of these tests (the mode
    # handler also starts the supervisor in an executor).
    monkeypatch.setattr("remote.tunnel.suspend", lambda: None)
    monkeypatch.setattr("remote.tunnel.resume", lambda: None)
    monkeypatch.setattr("remote.tunnel.start_background_supervisor", lambda: None)
    yield


# ─── the dispatchers all ask the same question ────────────────────────────
def test_every_dispatcher_follows_the_toggle(monkeypatch):
    from server.task_manager import TaskManager
    from tools.enhance import remote_enhance

    settings.save_address("10.0.0.5:50051")
    settings.set_worker_mode("local")
    assert TaskManager()._should_use_remote() is False
    assert remote_enhance.remote_enabled() is False

    settings.set_worker_mode("remote")
    assert TaskManager()._should_use_remote() is True
    assert remote_enhance.remote_enabled() is True


def test_on_the_fly_flip_applies_without_a_restart(monkeypatch):
    """The value is read per dispatch, so flipping it is enough."""
    from tools.enhance import remote_enhance
    settings.save_address("10.0.0.5:50051")
    assert remote_enhance.remote_enabled() is False
    settings.set_worker_mode("remote")
    assert remote_enhance.remote_enabled() is True


# ─── the API ───────────────────────────────────────────────────────────────
def _app(tmp_path, monkeypatch):
    from aiohttp import web
    from server import routes

    class FakeApp:
        def __init__(self):
            self.opt = types.SimpleNamespace(model="musetalk",
                                             remote_backend="musetalk",
                                             avatar_id="a")
            self.model_backend = ""
            self.model_state = "unloaded"

    fake = FakeApp()
    # The handler resolves the running app module (__main__ in production).
    # Patch it per test and let monkeypatch restore it afterwards.
    monkeypatch.setattr(routes, "_app_module", lambda: fake)
    app = web.Application()
    app["opt"] = fake.opt
    routes.setup_routes(app)
    return app, fake


def _call(app, method, path, **kw):
    from aiohttp.test_utils import TestClient, TestServer

    async def run():
        async with TestClient(TestServer(app)) as client:
            resp = await client.request(method, path, **kw)
            return resp.status, await resp.json()

    return asyncio.run(run())


def test_get_mode_reports_the_toggle(monkeypatch, tmp_path):
    app, _fake = _app(tmp_path, monkeypatch)
    status, body = _call(app, "GET", "/api/remote/mode")
    assert status == 200 and body["code"] == 0
    assert body["data"]["mode"] == "local"
    assert body["data"]["remote_active"] is False


def test_post_mode_persists_and_flips_opt_model(monkeypatch, tmp_path):
    app, fake = _app(tmp_path, monkeypatch)
    status, body = _call(app, "POST", "/api/remote/mode", json={"mode": "remote"})
    assert status == 200 and body["code"] == 0
    assert settings.worker_mode() == settings.MODE_REMOTE
    assert body["data"]["mode"] == "remote"
    # /api/admin/config reads opt.model: it must not still say musetalk.
    assert fake.opt.model == "remote"

    # A fresh app: an aiohttp Application is bound to the loop that served it.
    app2, fake2 = _app(tmp_path, monkeypatch)
    status, body = _call(app2, "POST", "/api/remote/mode", json={"mode": "local"})
    assert body["data"]["mode"] == "local"
    assert fake2.opt.model == "musetalk", "local mode needs a real local backend"


def test_post_mode_reports_a_resident_model_from_the_other_mode(monkeypatch,
                                                               tmp_path):
    app, fake = _app(tmp_path, monkeypatch)
    fake.model_backend = "remote"
    settings.set_worker_mode("remote")
    status, body = _call(app, "POST", "/api/remote/mode", json={"mode": "local"})
    assert body["data"]["reload_required"] is True
    assert body["data"]["resident_backend"] == "remote"


def test_post_mode_rejects_an_unknown_value(monkeypatch, tmp_path):
    app, _fake = _app(tmp_path, monkeypatch)
    status, body = _call(app, "POST", "/api/remote/mode", json={"mode": "gpu"})
    assert status == 200
    assert body["code"] == -1 and "local" in body["msg"]
    assert settings.worker_mode() == settings.MODE_LOCAL


def test_status_exposes_the_mode(monkeypatch, tmp_path):
    app, _fake = _app(tmp_path, monkeypatch)
    status, body = _call(app, "GET", "/api/remote/status")
    assert status == 200 and body["code"] == 0
    assert body["data"]["mode"] == "local"
    assert body["data"]["remote_active"] is False
