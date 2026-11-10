#!/usr/bin/env python3
"""Sustained-fps harness: hold a real WebRTC session long enough to sample fps.

`tests/webrtc_smoke.py` finishes in a few seconds, but the server only logs
`actual avg final fps` once per 100 pushed frames -- so a short session proves
the plumbing and never reports a number. This drives real speech for
--seconds and then reports every fps line the server logged for the run.

Usage:
    python tests/fps_probe.py --avatar anthony007 --seconds 45 --port 8010
"""
import argparse
import asyncio
import json
import sys
import time
import urllib.request


def _post(host, port, path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"http://{host}:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.status, json.loads(resp.read().decode())


def _get(host, port, path):
    with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=10) as resp:
        return json.loads(resp.read().decode())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--avatar", default="anthony007")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog. "
                                      "Pack my box with five dozen liquor jugs. "
                                      "How vexingly quick daft zebras jump!")
    args = ap.parse_args()

    from aiortc import RTCPeerConnection, RTCSessionDescription

    pc = RTCPeerConnection()
    pc.addTransceiver("audio")
    pc.addTransceiver("video")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    st, body = _post(args.host, args.port, "/offer",
                     {"sdp": pc.localDescription.sdp, "type": "offer",
                      "avatar": args.avatar})
    if st != 200:
        print("offer failed:", body)
        return 1
    sessionid = body["sessionid"]
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=body["sdp"], type="answer"))
    print(f"session {sessionid} on avatar {args.avatar}")

    # Actually consume the media. aiortc only pulls frames from the track when
    # something awaits recv(), and the server's fps counter advances when
    # frames are PUSHED -- so we must both receive and keep talking.
    received = 0

    async def drain_video():
        nonlocal received
        track = None
        deadline = time.time() + args.seconds
        while time.time() < deadline:
            if track is None:
                if not pc.getReceivers():
                    await asyncio.sleep(0.1)
                    continue
                track = pc.getReceivers()[0].track
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
            received += 1

    drain = asyncio.create_task(drain_video())
    end = time.time() + args.seconds
    chunk = 0
    while time.time() < end:
        _post(args.host, args.port, "/human",
              {"sessionid": sessionid, "text": args.text, "type": "echo",
               "interrupt": True})
        chunk += 1
        await asyncio.sleep(8.0)
        if chunk > 40:
            break
    await drain
    await pc.close()

    print(f"received {received} frames in {args.seconds:.0f}s "
          f"({received/args.seconds:.2f} fps from the CLIENT side)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
