#!/usr/bin/env python3
"""Render /admin.html with a NORMAL (non-building) session, stubbed.

Guards the refactor that moved the card body into a `details` variable: the
live path can't be exercised when the GPU worker is down, but the markup for a
real session (avatar id, model, voice ref, custom actions) must still render.

Usage:
    /opt/miniconda3/envs/livetalking/bin/python tests/admin_normal_card_ui.py
"""
import json
import os
import sys
import time

BASE = "http://127.0.0.1:8010"
SHOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "admin_ui_shots")

FAKE = {
    "code": 0, "msg": "ok",
    "data": {"sessions": [{
        "sessionid": "live-abc-999", "speaking": True, "recording": True,
        "building": False, "model": "remote", "avatar_id": "anthony007",
        "REF_FILE": "en-US-JennyNeural", "transport": "webrtc",
        "batch_size": 8, "customopt": [{"imgpath": "/x", "audiotype": 1}],
    }]},
}


def main():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    os.makedirs(SHOTS, exist_ok=True)
    results = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        results.append(ok)

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1500,1800")
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    d = webdriver.Chrome(options=opts)
    try:
        d.get(BASE + "/admin.html")
        d.execute_script("""
            const realFetch = window.fetch.bind(window);
            window.fetch = (url, opts) => String(url).includes('/api/admin/sessions')
                ? Promise.resolve(new Response(JSON.stringify(%s),
                    {headers: {'Content-Type': 'application/json'}}))
                : realFetch(url, opts);
        """ % json.dumps(FAKE))
        d.execute_script("fetchSessions()")
        time.sleep(1.5)

        text = d.find_element(By.ID, "sessionsContainer").text
        for label, needle in [
            ("session id", "live-abc-999"),
            ("transport", "webrtc"),
            ("avatar id", "anthony007"),
            ("model", "remote"),
            ("voice ref", "en-US-JennyNeural"),
            ("speaking badge", "Speaking"),
            ("recording badge", "Recording"),
            ("custom actions", "audiotype"),
        ]:
            check(f"renders {label}", needle in text, "" if needle in text else text[:150])

        check("no loading state for a live session", "Loading model" not in text)
        check("no undefined leaked", "undefined" not in text and "NaN" not in text)

        kill = d.find_elements(By.CSS_SELECTOR,
                               '.btn-kill-session[data-sessionid="live-abc-999"]')
        check("Kill button for the live session", len(kill) == 1)

        d.save_screenshot(os.path.join(SHOTS, "admin_normal_session.png"))

        errs = [l for l in d.get_log("browser")
                if l["level"] == "SEVERE" and "favicon.ico" not in l["message"]]
        check("no JS errors", not errs, str(errs[:2]))
    finally:
        d.quit()

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
