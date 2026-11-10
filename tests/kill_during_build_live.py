#!/usr/bin/env python3
"""Kill a session while its build is still running.

The first connect after an unload loads the model (tens of seconds) and its
registry slot is a bare ``None`` placeholder. There is no quit_event then, so
the kill must be remembered and ``create_session`` must discard the session
when the builder returns — otherwise the session reappears in the console and
its render threads are unreachable.

This needs the model UNLOADED first so the connect takes long enough to land a
kill inside the build window:

    curl -X POST localhost:8010/api/model/unload -d '{"force":true}'
    python tests/kill_during_build_live.py

Usage:
    /opt/miniconda3/envs/livetalking/bin/python tests/kill_during_build_live.py
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8010"


def _req(path, body=None, timeout=300):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def main():
    results = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        results.append(ok)

    st, body = _req("/api/model/status")
    print("model state:", body["data"]["state"])
    if body["data"]["state"] != "unloaded":
        print("refusing to run: unload the model first "
              "(curl -X POST localhost:8010/api/model/unload -d '{\"force\":true}')")
        return 1

    from aiortc import RTCPeerConnection
    import asyncio

    loop = asyncio.new_event_loop()

    async def offer():
        pc = RTCPeerConnection()
        pc.addTransceiver("audio")
        pc.addTransceiver("video")
        await pc.setLocalDescription(await pc.createOffer())
        return pc, pc.localDescription.sdp

    pc, sdp = loop.run_until_complete(offer())

    # Fire the connect in a thread: it blocks until the model is loaded.
    outcome = {}

    def do_offer():
        outcome["resp"] = _req("/offer", {"sdp": sdp, "type": "offer",
                                         "avatar": "anthony007"}, timeout=300)

    t = threading.Thread(target=do_offer, daemon=True)
    t.start()

    # Wait for the placeholder to appear, then kill it mid-build. The build
    # window is the model load (tens of seconds on the remote worker), during
    # which /api/admin/sessions reports the session with building=true.
    sid = None
    deadline = time.time() + 60
    while time.time() < deadline:
        st, body = _req("/api/admin/sessions")
        ids = [s["sessionid"] for s in body["data"]["sessions"] if s.get("building")]
        if ids:
            sid = ids[0]
            break
        time.sleep(0.2)

    if not sid:
        print("could not catch a session mid-build (model loaded too fast?)")
        t.join(timeout=300)
        loop.run_until_complete(pc.close())
        loop.close()
        return 1

    print("caught session mid-build:", sid)
    st, body = _req(f"/api/admin/sessions/{sid}/kill", {})
    check("kill accepted during build", body.get("code") == 0, str(body.get("msg")))

    st, body = _req("/api/admin/sessions")
    ids = [s["sessionid"] for s in body["data"]["sessions"]]
    check("session gone from console immediately", sid not in ids, str(ids))

    t.join(timeout=300)
    status, resp = outcome.get("resp", (None, {}))
    check("connect does not silently succeed", status != 200 or resp.get("sessionid") != sid,
          f"HTTP {status} {str(resp)[:80]}")

    # The decisive check: the session must NOT reappear once the build ends.
    time.sleep(2)
    st, body = _req("/api/admin/sessions")
    ids = [s["sessionid"] for s in body["data"]["sessions"]]
    check("killed session did not resurface after the build", sid not in ids, str(ids))
    check("active_sessions is 0", body["data"]["sessions"] == [], str(ids))

    loop.run_until_complete(pc.close())
    loop.close()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
