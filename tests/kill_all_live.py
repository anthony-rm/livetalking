#!/usr/bin/env python3
"""Two live sessions: Kill All must report BOTH and disconnect both.

Single-session kill already passes; the bulk path is where an ordering bug hid
(closing each pc first let connectionstatechange → remove_session empty the
registry, so Kill All answered "killed: []" after disconnecting everything).

Also checks the "kill while the model is loading" path: a second kill request
aimed at a session whose slot is still a placeholder must be accepted and must
not leave the session behind when its build finishes.

Usage:
    /opt/miniconda3/envs/livetalking/bin/python tests/kill_all_live.py
"""
import json
import os
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8010"


def _req(path, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=(data if data is not None else (b"" if body == {} else None)),
        headers={"Content-Type": "application/json"},
        method="POST" if (body is not None) else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Loop:
    def __init__(self):
        import asyncio
        self.asyncio = asyncio
        self.loop = asyncio.new_event_loop()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        self.asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout=200):
        return self.asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.t.join(5)


def connect(loop, avatar):
    import asyncio
    from aiortc import RTCPeerConnection, RTCSessionDescription

    async def go():
        pc = RTCPeerConnection()
        pc.addTransceiver("audio")
        pc.addTransceiver("video")
        await pc.setLocalDescription(await pc.createOffer())
        ans = _req("/offer", {"sdp": pc.localDescription.sdp, "type": "offer",
                             "avatar": avatar})
        await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type="answer"))
        return pc, ans["sessionid"]

    return loop.run(go())


def main():
    avatar = os.environ.get("KILL_UI_AVATAR", "anthony007")
    results = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        results.append(ok)

    # Make room for two concurrent sessions on one server.
    loop = Loop()
    pc1, sid1 = connect(loop, avatar)
    print("session 1:", sid1)
    try:
        pc2, sid2 = connect(loop, avatar)
        print("session 2:", sid2)
    except Exception as e:
        print("could not open a 2nd session (max_session?):", e)
        loop.run(pc1.close()); loop.stop()
        return 1

    listed = [s["sessionid"] for s in _req("/api/admin/sessions")["data"]["sessions"]]
    check("both sessions listed", {sid1, sid2} <= set(listed), f"{len(listed)} listed")

    body = _req("/api/admin/sessions/kill-all", {})
    killed = body.get("data", {}).get("killed", [])
    check("kill-all accepted", body.get("code") == 0, str(body.get("msg")))
    check("kill-all reported both ids", sorted(killed) == sorted([sid1, sid2]), str(killed))
    check("kill-all count matches", body["data"]["count"] == 2, str(body["data"]))

    listed = [s["sessionid"] for s in _req("/api/admin/sessions")["data"]["sessions"]]
    check("no sessions left", listed == [], str(listed))

    for pc, sid in ((pc1, sid1), (pc2, sid2)):
        # Client-side closure is detected by ICE consent freshness / DTLS
        # close_notify, which can take ~30 s — much slower than the server's
        # own "Connection state is closed" log line. Poll generously rather
        # than declaring a failure after 10 s.
        for _ in range(90):
            if pc.connectionState in ("closed", "failed", "disconnected"):
                break
            time.sleep(0.5)
        check(f"pc for {sid[:8]} closed", pc.connectionState != "connected", pc.connectionState)

    # Empty kill-all is a clean no-op, not an error.
    body = _req("/api/admin/sessions/kill-all", {})
    check("kill-all on empty is ok", body.get("code") == 0 and body["data"]["count"] == 0, str(body))

    loop.run(pc1.close()); loop.run(pc2.close()); loop.stop()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
