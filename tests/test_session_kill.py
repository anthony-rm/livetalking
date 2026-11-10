#!/usr/bin/env python3
"""Admin "kill session": the console's only way to end a live session.

Killing has TWO halves and both are tested here:

  * the avatar session (render / inference / TTS threads, GPU memory) — owned
    by ``SessionManager``; and
  * the ``RTCPeerConnection`` — closing only the session leaves the browser
    with a "connected" transceiver that never receives another frame, closing
    only the pc leaves the threads resident.

The tricky case is a session killed WHILE ITS BUILD IS RUNNING (the first
connect loads the model, which takes tens of seconds, and its slot is a bare
``None`` placeholder). There is no ``quit_event`` to set yet, so the kill is
recorded and ``create_session`` must discard the session when the builder
returns — otherwise the threads it starts later are unreachable and leak.

Run with the project interpreter:
    /opt/miniconda3/envs/livetalking/bin/python -m pytest tests/test_session_kill.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import routes  # noqa: E402
from server.session_manager import (  # noqa: E402
    SessionKilledError,
    SessionManager,
)


class FakeAvatar:
    """Just enough of BaseAvatar for the kill path."""

    def __init__(self, started=True):
        self.quit_event = threading.Event() if started else None
        self.flushed = False

    def flush_talk(self):
        self.flushed = True


@pytest.fixture()
def manager():
    """A private SessionManager; the module-level one is a singleton."""
    m = SessionManager.__new__(SessionManager)
    m.sessions = {}
    m.killed = set()
    m.build_session_fn = None
    m.max_session = 8
    m.initialized = True
    return m


# ─── SessionManager.kill_session ─────────────────────────────────────────────

def test_kill_live_session_sets_quit_event_and_unregisters(manager):
    avatar = FakeAvatar()
    manager.sessions["sid-1"] = avatar

    assert manager.kill_session("sid-1") is True

    assert avatar.flushed is True
    assert avatar.quit_event.is_set() is True
    assert "sid-1" not in manager.sessions
    assert manager.active_sessions() == 0


def test_kill_unknown_session_reports_not_found(manager):
    """Unknown id must be distinguishable — the API answers 404-ish, not ok."""
    assert manager.kill_session("nope") is False


def test_kill_session_before_render_does_not_crash(manager):
    """A built-but-not-started session has no quit_event; drop the reference."""
    avatar = FakeAvatar(started=False)
    manager.sessions["sid-2"] = avatar

    assert manager.kill_session("sid-2") is True
    assert "sid-2" not in manager.sessions


def test_kill_placeholder_records_intent(manager):
    """During the (slow) build the slot is None — there is nothing to stop yet."""
    manager.sessions["sid-3"] = None

    assert manager.kill_session("sid-3") is True
    assert "sid-3" not in manager.sessions
    assert manager.killed == {"sid-3"}


def test_kill_all_only_reports_real_sessions(manager):
    manager.sessions["a"] = FakeAvatar()
    manager.sessions["b"] = None          # still building
    manager.sessions["c"] = FakeAvatar()

    killed = manager.kill_all_sessions()

    assert sorted(killed) == ["a", "c"]
    assert manager.sessions == {}
    assert manager.killed == {"b"}


# ─── create_session must honour a kill that landed mid-build ──────────────────

def test_create_session_discards_a_session_killed_while_building(manager):
    """The builder publishes nothing; the caller must throw the result away.

    Without the `killed` check the session would be re-inserted into
    `sessions` after the admin already saw it disappear from the console, and
    its render threads would never be stopped.
    """
    built = FakeAvatar()

    def slow_build(sessionid, params):
        # The admin kills the placeholder while the model loads.
        assert manager.kill_session(sessionid) is True
        return built

    manager.build_session_fn = slow_build

    with pytest.raises(SessionKilledError):
        asyncio.run(manager.create_session({}, sessionid="sid-4"))

    assert "sid-4" not in manager.sessions
    assert manager.killed == set(), "the kill must be consumed exactly once"


def test_create_session_still_works_after_a_kill(manager):
    """Same sessionid retried later must not inherit the stale kill."""
    manager.build_session_fn = lambda sid, params: FakeAvatar()

    asyncio.run(manager.create_session({}, sessionid="sid-5"))
    assert "sid-5" in manager.sessions
    # A kill, then a fresh connect reusing the id (WHEP lets clients pick it).
    manager.kill_session("sid-5")
    asyncio.run(manager.create_session({}, sessionid="sid-5"))
    assert manager.sessions["sid-5"] is not None


def test_failed_build_does_not_leak_a_placeholder(manager):
    """A model-load failure must not leave an invisible, unkillable slot.

    The placeholder is excluded from /api/admin/sessions (it has no opts) and
    no code path removed it, so with max_session=1 a single failed connect
    bricked every later connect while the console showed nothing.
    """
    manager.max_session = 1

    def boom(sid, params):
        raise RuntimeError("worker unreachable")

    manager.build_session_fn = boom
    with pytest.raises(RuntimeError):
        asyncio.run(manager.create_session({}, sessionid="sid-broke"))

    assert manager.sessions == {}, "the placeholder must be removed on failure"

    # The slot is usable again.
    manager.build_session_fn = lambda sid, params: FakeAvatar()
    asyncio.run(manager.create_session({}, sessionid="sid-retry"))
    assert manager.sessions["sid-retry"] is not None


def test_failed_build_after_a_kill_clears_the_pending_kill(manager):
    """A kill that races a failing build must not poison the next attempt."""
    def boom(sid, params):
        raise RuntimeError("worker unreachable")

    manager.build_session_fn = boom
    manager.sessions["sid-z"] = None
    manager.kill_session("sid-z")
    with pytest.raises(RuntimeError):
        asyncio.run(manager.create_session({}, sessionid="sid-z"))

    assert manager.killed == set()
    assert manager.sessions == {}


# ─── HTTP surface ────────────────────────────────────────────────────────────

class FakeRTC:
    """Records which peer connections were asked to close."""

    def __init__(self):
        self.closed = []

    async def close_session(self, sessionid):
        self.closed.append(sessionid)
        return True


class RaceRTC(FakeRTC):
    """A manager whose close_session mimics the real on_connectionstatechange.

    Closing a pc flips it to "closed", and RTCManager's handler calls
    ``remove_session``. If the endpoint closed the pc BEFORE killing the
    session, that side effect unregistered the session and the kill reported
    "not found" for a session it had just successfully disconnected.
    """

    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    async def close_session(self, sessionid):
        await super().close_session(sessionid)
        self.manager.remove_session(sessionid)
        return True


def _app_with(rtc):
    from aiohttp import web

    app = web.Application()
    app["rtc_manager"] = rtc
    app["opt"] = types.SimpleNamespace(model="wavemesh")
    routes.setup_routes(app)
    return app


def _call(app, method, path, **kw):
    """Drive one request through a real aiohttp server (no test plugin needed)."""
    from aiohttp.test_utils import TestClient, TestServer

    async def run():
        async with TestClient(TestServer(app)) as client:
            assert client.server is not None
            resp = await client.request(method, path, **kw)
            return resp.status, await resp.json()

    return asyncio.run(run())


def test_kill_endpoint_closes_pc_then_session(monkeypatch, manager):
    avatar = FakeAvatar()
    manager.sessions["sid-6"] = avatar
    rtc = FakeRTC()
    monkeypatch.setattr(routes, "session_manager", manager)

    status, body = _call(_app_with(rtc), "POST", "/api/admin/sessions/sid-6/kill")

    assert status == 200
    assert body["code"] == 0
    assert body["data"]["killed"] is True
    assert rtc.closed == ["sid-6"], "the peer connection must be closed too"
    assert avatar.quit_event.is_set() is True
    assert "sid-6" not in manager.sessions


def test_kill_endpoint_404s_like_for_unknown_session(monkeypatch, manager):
    """A stale console (the card was already gone) must not report success."""
    monkeypatch.setattr(routes, "session_manager", manager)

    status, body = _call(_app_with(FakeRTC()), "POST", "/api/admin/sessions/ghost/kill")

    assert status == 200                      # JSON body, not an HTTP error
    assert body["code"] == -1
    assert "not found" in body["msg"]


def test_kill_all_endpoint_reports_what_it_killed(monkeypatch, manager):
    manager.sessions["x"] = FakeAvatar()
    manager.sessions["y"] = FakeAvatar()
    rtc = FakeRTC()
    monkeypatch.setattr(routes, "session_manager", manager)

    status, body = _call(_app_with(rtc), "POST", "/api/admin/sessions/kill-all")

    assert status == 200
    assert sorted(body["data"]["killed"]) == ["x", "y"]
    assert body["data"]["count"] == 2
    assert sorted(rtc.closed) == ["x", "y"]
    assert body["data"]["active_sessions"] == 0


def test_kill_endpoint_survives_the_close_side_effect(monkeypatch, manager):
    """Closing the pc unregisters the session; the kill must still say "ok"."""
    avatar = FakeAvatar()
    manager.sessions["sid-7"] = avatar
    rtc = RaceRTC(manager)
    monkeypatch.setattr(routes, "session_manager", manager)

    status, body = _call(_app_with(rtc), "POST", "/api/admin/sessions/sid-7/kill")

    assert body["code"] == 0, body
    assert body["data"]["killed"] is True
    assert avatar.quit_event.is_set() is True


def test_kill_all_reports_the_sessions_it_killed(monkeypatch, manager):
    """Same race for the bulk path — it must not answer "killed: []"."""
    manager.sessions["p"] = FakeAvatar()
    manager.sessions["q"] = FakeAvatar()
    rtc = RaceRTC(manager)
    monkeypatch.setattr(routes, "session_manager", manager)

    status, body = _call(_app_with(rtc), "POST", "/api/admin/sessions/kill-all")

    assert body["code"] == 0, body
    assert sorted(body["data"]["killed"]) == ["p", "q"], body


def test_admin_sessions_lists_a_building_session(monkeypatch, manager):
    """A session that is loading the model must be visible AND killable.

    It occupies a max_session slot for tens of seconds while invisible in the
    console — exactly when an admin needs to see and kill it.
    """
    manager.sessions["building-1"] = None
    manager.sessions["live-1"] = FakeAvatar()
    manager.sessions["live-1"].opt = types.SimpleNamespace(
        model="remote", avatar_id="a1", REF_FILE="", transport="webrtc",
        batch_size=8, customopt=[])
    manager.sessions["live-1"].recording = False
    monkeypatch.setattr(routes, "session_manager", manager)

    status, body = _call(_app_with(FakeRTC()), "GET", "/api/admin/sessions")

    assert body["code"] == 0
    by_id = {s["sessionid"]: s for s in body["data"]["sessions"]}
    assert set(by_id) == {"building-1", "live-1"}
    assert by_id["building-1"]["building"] is True
    assert by_id["live-1"]["building"] is False
    assert by_id["live-1"]["avatar_id"] == "a1"


def test_kill_endpoint_accepts_a_building_session(monkeypatch, manager):
    """The console shows it, so the Kill button must work on it too."""
    manager.sessions["building-2"] = None
    monkeypatch.setattr(routes, "session_manager", manager)

    status, body = _call(_app_with(FakeRTC()), "POST",
                         "/api/admin/sessions/building-2/kill")

    assert body["code"] == 0, body
    assert manager.sessions == {}
    assert manager.killed == {"building-2"}


def test_kill_endpoint_url_decodes_client_chosen_ids(monkeypatch, manager):
    """WHEP lets the CLIENT name its sessionid, so ids are arbitrary strings."""
    weird = "sess with space/%2F"
    avatar = FakeAvatar()
    manager.sessions[weird] = avatar
    monkeypatch.setattr(routes, "session_manager", manager)

    from urllib.parse import quote
    status, body = _call(_app_with(FakeRTC()), "POST",
                         f"/api/admin/sessions/{quote(weird, safe='')}/kill")

    assert body["code"] == 0, body
    assert manager.sessions == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
