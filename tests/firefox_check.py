#!/usr/bin/env python3
"""Drive the REAL installed Firefox against LiveTalking and report whether
the video actually renders. Uses Selenium's WebDriver BiDi backend.

Usage:
    python tests/firefox_check.py --port 8010
"""
import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--url", default=None)
    args = ap.parse_args()
    url = args.url or f"http://127.0.0.1:{args.port}/index.html"

    from selenium import webdriver
    from selenium.webdriver.common.by import By

    opts = webdriver.FirefoxOptions()
    opts.binary_location = "/Applications/Firefox.app/Contents/MacOS/firefox"
    opts.add_argument("--width=1280")
    opts.add_argument("--height=900")
    # BiDi needs an explicit websocket URL for the script module
    opts.set_capability("webSocketUrl", True)

    driver = webdriver.Firefox(options=opts)
    try:
        # Capture the peer connection before the page script runs
        driver.script.add_preload_script("""
            const Orig = window.RTCPeerConnection;
            window.__pcs = [];
            window.RTCPeerConnection = function (...args) {
                const pc = new Orig(...args);
                window.__pcs.push(pc);
                return pc;
            };
            window.RTCPeerConnection.prototype = Orig.prototype;
        """)
        print("navigating to", url)
        driver.get(url)
        time.sleep(2)

        print("start() defined:", driver.execute_script("return typeof window.start"))

        btn = driver.find_element(By.ID, "btnStart")
        print("clicking 开始连接 ...")
        btn.click()
        time.sleep(10)

        state = driver.execute_script("""
            const v = document.getElementById('video');
            const a = document.getElementById('audio');
            return {
              btnStart: document.getElementById('btnStart').style.display,
              btnStop: document.getElementById('btnStop').style.display,
              badge: document.getElementById('statusBadge').innerText,
              sid: document.getElementById('sessionIdDisplay').innerText,
              errorText: document.getElementById('offerError').innerText,
              videoTracks: v && v.srcObject
                  ? v.srcObject.getTracks().map(t => t.kind + ':' + t.readyState).join(',')
                  : 'NULL',
              videoW: v ? v.videoWidth : null,
              videoH: v ? v.videoHeight : null,
              videoReadyState: v ? v.readyState : null,
              videoPaused: v ? v.paused : null,
              audioReadyState: a ? a.readyState : null,
              audioPaused: a ? a.paused : null,
            };
        """)
        print("--- state after click (real Firefox) ---")
        print(json.dumps(state, indent=2, ensure_ascii=False))

        # Pull Firefox's own RTP statistics
        stats = driver.execute_async_script("""
            const done = arguments[arguments.length - 1];
            // index.html declares `let pc` at script scope -> global lexical
            // binding, NOT a property of window. Reference it directly.
            const pcRef = (typeof pc !== 'undefined') ? pc : (window.__pcs && window.__pcs[0]);
            if (!pcRef) return done({error: 'no pc found'});
            pcRef.getStats().then(report => {
              const out = {inbound: [], codecs: [], receivers: []};
              for (const [, st] of report) {
                if (st.type === 'inbound-rtp') {
                  out.inbound.push({
                    kind: st.kind,
                    packets: st.packetsReceived,
                    bytes: st.bytesReceived,
                    framesDecoded: st.framesDecoded,
                    framesDropped: st.framesDropped,
                    decoder: st.decoderImplementation,
                    codecId: st.codecId,
                    nackCount: st.nackCount,
                    pliCount: st.pliCount,
                  });
                }
                if (st.type === 'codec') {
                  out.codecs.push({id: st.id, mime: st.mimeType, fmtp: st.sdpFmtpLine});
                }
                if (st.type === 'receiver') {
                  out.receivers.push({
                    kind: st.kind, codecId: st.codecId,
                    trackIdentifier: st.trackIdentifier,
                  });
                }
              }
              done(out);
            }).catch(e => done({error: String(e)}));
        """)
        print("--- Firefox getStats ---")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

        ok = state.get("videoW") or 0
        print("\nRESULT:", "FIREFOX VIDEO OK" if ok else "FIREFOX VIDEO BLANK")
        return 0 if ok else 1
    finally:
        driver.quit()


if __name__ == "__main__":
    raise SystemExit(main())