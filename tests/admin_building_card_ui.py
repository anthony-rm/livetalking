#!/usr/bin/env python3
"""Render /admin.html with a session that is still loading the model.

The live worker is down right now, so this stubs only the /api/admin/sessions
response in the browser (everything else is fetched for real) and checks the
"Loading model..." card renders with a working Kill button instead of blank
"undefined" fields. Any exception inside renderSessions would leave the
container empty or log a SEVERE error.

Usage:
    /opt/miniconda3/envs/livetalking/bin/python tests/admin_building_card_ui.py
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
    "data": {"sessions": [
        {"sessionid": "building-abc-123", "speaking": False,
         "recording": False, "building": True},
    ]},
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
        # Stub only the sessions endpoint, AFTER the page's own fetch runs.
        d.execute_script("""
            const realFetch = window.fetch.bind(window);
            window.fetch = (url, opts) => {
                if (String(url).includes('/api/admin/sessions') ) {
                    return Promise.resolve(new Response(JSON.stringify(%s),
                        {headers: {'Content-Type': 'application/json'}}));
                }
                return realFetch(url, opts);
            };
        """ % json.dumps(FAKE))
        d.execute_script("fetchSessions()")
        time.sleep(1.5)

        cards = d.find_elements(By.CSS_SELECTOR, ".session-card")
        check("a card rendered", len(cards) == 1, f"{len(cards)} cards")

        body = d.find_element(By.ID, "sessionsContainer").text
        check("shows the loading state", "Loading model" in body, body.replace("\n", " | ")[:120])
        check("no undefined/NaN leaked", "undefined" not in body and "NaN" not in body,
              body.replace("\n", " | ")[:120])

        kill = d.find_elements(By.CSS_SELECTOR,
                               '.btn-kill-session[data-sessionid="building-abc-123"]')
        check("Kill button present for the building session", len(kill) == 1)
        check("Kill button enabled", kill[0].is_enabled() if kill else False)

        check("Kill All enabled", d.find_element(By.ID, "btnKillAll").is_enabled())

        d.save_screenshot(os.path.join(SHOTS, "admin_building_session.png"))

        errs = [l for l in d.get_log("browser")
                if l["level"] == "SEVERE" and "favicon.ico" not in l["message"]]
        check("no JS errors", not errs, str(errs[:2]))
    finally:
        d.quit()

    print(f"\n{sum(results)}/{len(results)} checks passed")
    print("screenshot:", os.path.join(SHOTS, "admin_building_session.png"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
