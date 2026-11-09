#!/usr/bin/env python3
"""Test whether disabling Firefox's mDNS candidate obfuscation fixes ICE.

Firefox publishes local host candidates as <uuid>.local. aioice (LiveTalking's
ICE stack) must resolve those via multicast DNS; on this machine that fails
(likely VPN TUN interference), so the server's ICE collapses.

Pref: media.peerconnection.mdns_control = false -> Firefox sends real IPs.

Usage: python ff_mdns_pref.py [port] [true|false]
"""
import json
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8011
MDNS = sys.argv[2] if len(sys.argv) > 2 else "false"
URL = f"http://127.0.0.1:{PORT}/index.html"

from selenium import webdriver
from selenium.webdriver.common.by import By

opts = webdriver.FirefoxOptions()
opts.binary_location = "/Applications/Firefox.app/Contents/MacOS/firefox"
opts.set_capability("webSocketUrl", True)
# Disable mDNS obfuscation so WebRTC candidates carry real IPs
opts.set_preference("media.peerconnection.mdns_control", MDNS)

print(f"media.peerconnection.mdns_control = {MDNS}")
driver = webdriver.Firefox(options=opts)
try:
    driver.get(URL)
    time.sleep(2)
    driver.find_element(By.ID, "btnStart").click()
    print("clicked 开始连接; waiting 15s...")
    time.sleep(15)

    state = driver.execute_script("""
        const v = document.getElementById('video');
        const a = document.getElementById('audio');
        return {
          badge: document.getElementById('statusBadge').innerText,
          sid: document.getElementById('sessionIdDisplay').innerText,
          videoW: v.videoWidth, videoH: v.videoHeight,
          videoReady: v.readyState, videoPaused: v.paused,
          audioReady: a.readyState,
          tracks: v.srcObject ? v.srcObject.getTracks().map(t => t.kind+':'+t.readyState).join(',') : 'NULL',
        };
    """)
    print("--- Firefox state ---")
    print(json.dumps(state, indent=2, ensure_ascii=False))
    ok = bool(state.get("videoW"))
    print("\nRESULT:", "FIREFOX VIDEO OK" if ok else "FIREFOX VIDEO BLANK")
    sys.exit(0 if ok else 1)
finally:
    driver.quit()