#!/usr/bin/env python3
"""LiveTalking HTTP/API smoke test.

Usage:
    python tests/smoke_test.py [--host 127.0.0.1] [--port 8010]

Probes the endpoints documented in docs/api.md without needing model
weights or a WebRTC session. Works against a server started with
--skip-model as well as a full one.

Exit code: 0 if all checks pass, 1 otherwise.
"""
import argparse
import json
import sys
import urllib.request
import urllib.error

def _req(base, method, path, body=None, headers=None):
    url = f"{base}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except urllib.error.URLError as e:
        return None, str(e)

def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    results = []

    # 1. Root redirect
    st, body = _req(base, "GET", "/")
    results.append(check("GET / responds", st is not None and st in (200, 302), f"HTTP {st}"))

    # 2. Static web page served
    st, body = _req(base, "GET", "/index.html")
    results.append(check("GET /index.html", st == 200, f"HTTP {st}"))

    # 3. Admin config
    st, body = _req(base, "GET", "/api/admin/config")
    ok = False
    if st == 200:
        try:
            ok = json.loads(body).get("code") == 0
        except Exception:
            pass
    results.append(check("GET /api/admin/config", ok, f"HTTP {st}"))

    # 4. Admin sessions
    st, body = _req(base, "GET", "/api/admin/sessions")
    ok = False
    if st == 200:
        try:
            ok = json.loads(body).get("code") == 0
        except Exception:
            pass
    results.append(check("GET /api/admin/sessions", ok, f"HTTP {st}"))

    # 5. Avatar tasks list
    st, body = _req(base, "GET", "/api/avatar/tasks")
    ok = False
    if st == 200:
        try:
            ok = json.loads(body).get("code") == 0
        except Exception:
            pass
    results.append(check("GET /api/avatar/tasks", ok, f"HTTP {st}"))

    # 6. POST /human without session → graceful error
    st, body = _req(base, "POST", "/human",
                    {"sessionid": "smoke-test-ghost", "text": "hello", "type": "echo"})
    ok = st == 200
    results.append(check("POST /human (no session)", ok, f"HTTP {st}"))

    # 7. POST /is_speaking without session → graceful error
    st, body = _req(base, "POST", "/is_speaking",
                    {"sessionid": "smoke-test-ghost"})
    ok = st == 200
    results.append(check("POST /is_speaking (no session)", ok, f"HTTP {st}"))

    # 8. POST /interrupt_talk without session → graceful error
    st, body = _req(base, "POST", "/interrupt_talk",
                    {"sessionid": "smoke-test-ghost"})
    ok = st == 200
    results.append(check("POST /interrupt_talk (no session)", ok, f"HTTP {st}"))

    total = len(results)
    passed = sum(results)
    print(f"\n{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()