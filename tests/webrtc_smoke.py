#!/usr/bin/env python3
"""End-to-end WebRTC smoke test for LiveTalking.

Creates a real aiortc PeerConnection, POSTs the SDP offer to /offer,
gets the answer back, and confirms a session exists. Then drives the
avatar with text via /human and polls /is_speaking.

Usage:
    python tests/webrtc_smoke.py [--host 127.0.0.1] [--port 8010]

Requires: pip install aiortc (already in the livetalking env)
"""
import argparse
import asyncio
import json
import sys
import urllib.request
import urllib.error

def _post_json(host, port, path, body):
    url = f"http://{host}:{port}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, json.loads(resp.read().decode())

def _get_json(host, port, path):
    url = f"http://{host}:{port}{path}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    from aiortc import RTCPeerConnection, RTCSessionDescription

    results = []
    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        results.append(ok)

    # 1. Create a peer connection and generate an offer
    pc = RTCPeerConnection()
    pc.addTransceiver("audio")
    pc.addTransceiver("video")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    sdp = pc.localDescription.sdp

    # 2. POST the offer to /offer
    st, body = _post_json(args.host, args.port, "/offer",
                          {"sdp": sdp, "type": "offer",
                           "avatar": "wav2lip256_avatar1"})
    check("POST /offer returns 200", st == 200, f"HTTP {st}")
    if st != 200:
        print("Body:", body)
        sys.exit(1)

    sessionid = body.get("sessionid")
    check("sessionid returned", bool(sessionid), sessionid or "missing")
    check("answer SDP present", bool(body.get("sdp")), "sdp len=" + str(len(body.get("sdp", ""))))

    # 3. Set the remote description (completes the handshake)
    answer = RTCSessionDescription(sdp=body["sdp"], type="answer")
    await pc.setRemoteDescription(answer)
    await asyncio.sleep(1)

    # 4. Confirm session shows up in admin
    st, body = _get_json(args.host, args.port, "/api/admin/sessions")
    sessions = body.get("data", {}).get("sessions", [])
    check("session appears in admin", any(s.get("sessionid") == sessionid for s in sessions),
          f"{len(sessions)} sessions")

    # 5. Drive the avatar with text (echo mode)
    st, body = _post_json(args.host, args.port, "/human",
                          {"sessionid": sessionid, "text": "Hello LiveTalking", "type": "echo"})
    check("POST /human accepted", st == 200 and body.get("code") == 0, f"HTTP {st} code={body.get('code')}")

    # 6. Query speaking state
    st, body = _post_json(args.host, args.port, "/is_speaking",
                          {"sessionid": sessionid})
    speaking = body.get("data")
    check("POST /is_speaking returns bool", isinstance(speaking, bool), str(speaking))

    # 7. Interrupt
    st, body = _post_json(args.host, args.port, "/interrupt_talk",
                          {"sessionid": sessionid})
    check("POST /interrupt_talk accepted", st == 200 and body.get("code") == 0, f"HTTP {st}")

    # 8. Close the peer connection (session cleanup is async — give the
    #    connectionstatechange handler a moment to fire)
    await pc.close()
    for _ in range(10):
        await asyncio.sleep(0.5)
        st, body = _get_json(args.host, args.port, "/api/admin/sessions")
        sessions = body.get("data", {}).get("sessions", [])
        if not any(s.get("sessionid") == sessionid for s in sessions):
            break
    st, body = _get_json(args.host, args.port, "/api/admin/sessions")
    sessions = body.get("data", {}).get("sessions", [])
    check("session cleaned up after close",
          not any(s.get("sessionid") == sessionid for s in sessions),
          f"{len(sessions)} sessions remain")

    total = len(results)
    passed = sum(results)
    print(f"\n{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    asyncio.run(main())