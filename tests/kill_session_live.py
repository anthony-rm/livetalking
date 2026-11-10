#!/usr/bin/env python3
"""Live check for the admin kill endpoint against a RUNNING server.

Creates a real aiortc PeerConnection, connects through /offer, then kills the
session over HTTP and asserts both halves died: the session vanished from
/api/admin/sessions AND the browser-side transceiver stopped receiving
(a closed pc makes recv() raise / return nothing).

Usage:
    /opt/miniconda3/envs/livetalking/bin/python tests/kill_session_live.py \
        [--host 127.0.0.1] [--port 8010] [--avatar wav2lip256_avatar1]
"""
import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request


def _req(host, port, path, body=None):
    url = f"http://{host}:{port}{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--avatar", default="wav2lip256_avatar1")
    ap.add_argument("--connect-timeout", type=float, default=180.0)
    args = ap.parse_args()

    from aiortc import RTCPeerConnection

    results = []
    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        results.append(ok)

    pc = RTCPeerConnection()
    pc.addTransceiver("audio")
    pc.addTransceiver("video")
    await pc.setLocalDescription(await pc.createOffer())

    st, body = _req(args.host, args.port, "/offer", {
        "sdp": pc.localDescription.sdp, "type": "offer", "avatar": args.avatar})
    if st != 200 or body.get("code") not in (0, None) or not body.get("sessionid"):
        print("offer rejected:", st, body)
        await pc.close()
        return 1
    sessionid = body["sessionid"]

    # Register track handlers BEFORE setRemoteDescription: aiortc fires the
    # "track" event as soon as the answer is applied, so a handler attached
    # afterwards misses the track and the test sees zero frames.
    got_video = []

    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            return
        async def drain():
            try:
                while True:
                    await track.recv()
                    got_video.append(1)
            except Exception:
                pass
        asyncio.ensure_future(drain())

    from aiortc import RTCSessionDescription
    await pc.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type="answer"))
    check("session created", True, sessionid)

    deadline = asyncio.get_event_loop().time() + args.connect_timeout
    while not got_video and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.5)
    check("video frames flowing before kill", bool(got_video),
          f"{len(got_video)} frames, pc={pc.connectionState}")

    st, body = _req(args.host, args.port, "/api/admin/sessions")
    listed = [s["sessionid"] for s in body.get("data", {}).get("sessions", [])]
    check("session listed in console", sessionid in listed, f"{listed}")

    # The kill.
    st, body = _req(args.host, args.port,
                    f"/api/admin/sessions/{sessionid}/kill", {})
    check("POST kill accepted", st == 200 and body.get("code") == 0, str(body.get("msg")))

    st, body = _req(args.host, args.port, "/api/admin/sessions")
    listed = [s["sessionid"] for s in body.get("data", {}).get("sessions", [])]
    check("session gone from console", sessionid not in listed, f"{listed}")

    # The remote pc must notice the close, not hang "connected" forever.
    for _ in range(20):
        if pc.connectionState in ("closed", "failed", "disconnected"):
            break
        await asyncio.sleep(0.5)
    check("peer connection closed/failed", pc.connectionState in ("closed", "failed", "disconnected"),
          pc.connectionState)

    n_before = len(got_video)
    await asyncio.sleep(3)
    check("frames stopped after kill", len(got_video) == n_before,
          f"{n_before} -> {len(got_video)}")

    st, body = _req(args.host, args.port, "/api/model/status")
    check("active_sessions back to 0", body["data"]["active_sessions"] == 0,
          str(body["data"]))

    # Killing again is a clean "not found", not a 500.
    st, body = _req(args.host, args.port,
                    f"/api/admin/sessions/{sessionid}/kill", {})
    check("second kill reports not found", body.get("code") == -1, str(body.get("msg")))

    await pc.close()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
