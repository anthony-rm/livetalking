#!/usr/bin/env python3
"""Check whether actual media frames flow over WebRTC from LiveTalking.

Handshakes with /offer, then counts video and audio frames received on the
remote tracks. Also reports the negotiated codec for each kind.
"""
import asyncio
import json
import urllib.request
import sys

from aiortc import RTCPeerConnection, RTCSessionDescription


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    pc = RTCPeerConnection()
    stats = {"video": 0, "audio": 0}
    codecs = {}
    first_frame = asyncio.Event()

    @pc.on("track")
    def on_track(track):
        kind = track.kind
        print(f"  track received: kind={kind}")

        @track.on("frame")
        def on_frame(frame):
            if stats[kind] == 0:
                first_frame.set()
            stats[kind] += 1

    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    body = json.dumps({"sdp": pc.localDescription.sdp, "type": "offer",
                       "avatar": "wav2lip256_avatar1"}).encode()
    req = urllib.request.Request(f"{base}/offer", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        ans = json.loads(r.read().decode())
    print("session:", ans.get("sessionid"))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type="answer"))

    # Let ICE settle
    for _ in range(50):
        if pc.connectionState in ("connected", "failed"):
            break
        await asyncio.sleep(0.2)
    print("connectionState:", pc.connectionState)

    # Report negotiated codecs from the answer SDP
    for line in ans["sdp"].splitlines():
        if line.startswith("a=rtpmap:") and any(
                c in line for c in ("H264", "VP8", "opus", "PCMU", "PCMA")):
            codecs.setdefault(line.split()[1].split("/")[0], []).append(
                line.split()[1].split("/")[0])
    print("negotiated rtpmap lines:")
    for line in ans["sdp"].splitlines():
        if line.startswith("a=rtpmap:"):
            print("   ", line)

    # Count frames for up to 20s
    print("waiting for frames (20s max)...")
    deadline = asyncio.get_event_loop().time() + 20
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.5)
        if stats["video"] >= 10 and stats["audio"] >= 10:
            break
    print(f"\nframes received: video={stats['video']} audio={stats['audio']}")

    # Report inbound RTP stats
    report = await pc.getStats()
    for _, stat in report.items():
        if stat.type == "inbound-rtp":
            print(f"  inbound-rtp {stat.kind}: packets={stat.packetsReceived} "
                  f"bytes={getattr(stat,'bytesReceived','?')} "
                  f"frames={getattr(stat,'framesDecoded','?')} "
                  f"decoder={getattr(stat,'decoderImplementation','?')}")

    await pc.close()
    ok = stats["video"] > 0 and stats["audio"] > 0
    print("\nMEDIA OK" if ok else "\nNO MEDIA FLOWING")
    sys.exit(0 if ok else 1)


asyncio.run(main())