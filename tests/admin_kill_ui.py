#!/usr/bin/env python3
"""Drive /admin.html in headless Chrome: does the Kill control actually work?

Renders the page against the LIVE server, checks the per-session Kill button
and the Kill All button exist and reach the endpoints, and screenshots the
result. Uses a real session created through /offer so there is something to
kill.

Usage:
    /opt/miniconda3/envs/livetalking/bin/python tests/admin_kill_ui.py
"""
import asyncio
import json
import os
import sys
import urllib.request

BASE = "http://127.0.0.1:8010"
SHOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "admin_ui_shots")


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return json.loads(r.read().decode())


def _post(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data or b"",
                                headers={"Content-Type": "application/json"},
                                method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


async def make_session():
    from aiortc import RTCPeerConnection, RTCSessionDescription
    pc = RTCPeerConnection()
    pc.addTransceiver("audio")
    pc.addTransceiver("video")
    await pc.setLocalDescription(await pc.createOffer())
    body = json.loads(json.dumps({
        "sdp": pc.localDescription.sdp, "type": "offer",
        "avatar": os.environ.get("KILL_UI_AVATAR", "anthony007")}))
    req = urllib.request.Request(BASE + "/offer", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        ans = json.loads(r.read().decode())
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type="answer"))
    return pc, ans["sessionid"]


class LoopThread:
    """Own an event loop in a background thread for the whole UI run.

    aiortc's ICE agent keeps tasks on the loop that created the pc. Driving
    that loop with a bare run_until_complete() leaves those tasks unscheduled
    the moment it returns (the agent then dies on `None.call_exception_handler`
    and, worse, the server sees the connection fail and drops the session
    before the browser can click Kill). Keep the loop spinning instead.
    """

    def __init__(self):
        import threading
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        try:
            self.loop.close()
        except Exception:
            pass


def main():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    os.makedirs(SHOTS, exist_ok=True)
    loop = LoopThread()
    pc, sid = loop.run(make_session(), timeout=200)
    print("live session:", sid)

    results = []
    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        results.append(ok)

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1500,2400")
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(BASE + "/admin.html")
        wait = WebDriverWait(driver, 20)

        # The card for the live session, with its Kill button.
        sel = f'.btn-kill-session[data-sessionid="{sid}"]'
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
        check("per-session Kill button rendered", True)

        kill_all = driver.find_element(By.ID, "btnKillAll")
        check("Kill All enabled with a live session", kill_all.is_enabled())

        n = len(driver.find_elements(By.CSS_SELECTOR, ".btn-kill-session"))
        check("one Kill button per session", n == 1, f"{n} buttons")

        driver.save_screenshot(os.path.join(SHOTS, "admin_with_session.png"))

        # Errors in the inline script surface here, not in a screenshot.
        # favicon.ico is a missing static asset, not a script error.
        logs = [l for l in driver.get_log("browser")
                if not l["message"].startswith(BASE + "/favicon.ico")]
        errors = [l for l in logs if l["level"] == "SEVERE"]
        check("no JS errors on load", not errors, str(errors[:2]))

        # Click it (confirm() is auto-accepted by Chrome's default behaviour
        # only if we handle the dialog; clicking via JS bypasses the prompt
        # but still exercises the fetch + re-render path).
        driver.execute_script(
            "window.confirm = () => true;"
            f"document.querySelector('{sel}').click();")

        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#sessionAlert .alert")))
        alert_text = driver.find_element(By.CSS_SELECTOR, "#sessionAlert").text
        check("kill reports an outcome", bool(alert_text), alert_text.strip())
        check("confirmation names the session", "killed" in alert_text.lower(), alert_text.strip())

        wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, ".btn-kill-session")) == 0)
        check("session card removed after kill", True)
        check("Kill All disabled when empty", not driver.find_element(By.ID, "btnKillAll").is_enabled())

        driver.save_screenshot(os.path.join(SHOTS, "admin_after_kill.png"))

        sessions = _get("/api/admin/sessions")["data"]["sessions"]
        check("server has no sessions left", sessions == [], str(sessions))

        logs = driver.get_log("browser")
        errors = [l for l in logs if l["level"] == "SEVERE"
                  and not l["message"].startswith(BASE + "/favicon.ico")]
        check("no JS errors after kill", not errors, str(errors[:2]))
    finally:
        driver.quit()
        try:
            loop.run(pc.close(), timeout=10)
        except Exception:
            pass
        loop.stop()

    print(f"\n{sum(results)}/{len(results)} checks passed")
    print("screenshots:", SHOTS)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
