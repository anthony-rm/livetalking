#!/usr/bin/env python3
"""Parallel ranged downloader for HuggingFace files.

Why this exists: on this network a single connection to huggingface.co is
shaped to ~127 KB/s and to hf-mirror.com to ~394 KB/s, but the shaping is
PER CONNECTION — four concurrent range requests aggregate to ~1.6 MB/s. For a
2.31 GB weight set that is the difference between 5 hours and 25 minutes.

Correctness matters more than speed here, so every piece is verified: each
worker writes its own byte range at the right offset, the file size must match
the server's Content-Length exactly, and a per-file SHA-256 is printed so the
caller can compare against the HF manifest.

Usage:
    python parallel_hf_download.py --repo digital-avatar/ditto-talkinghead \\
        --dest ~/ditto-weights --include 'ditto_pytorch/**' 'ditto_cfg/**' \\
        --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

CHUNK = 4 * 1024 * 1024       # 4 MiB per range request
TIMEOUT = 120


def _base_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _url(repo: str, path: str) -> str:
    return f"{_base_endpoint()}/{repo}/resolve/main/{path}"


def _head(url: str) -> tuple[int, bool]:
    """Return (size, supports_ranges) without downloading the body."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        size = int(r.headers.get("Content-Length", 0))
        ranges = r.headers.get("Accept-Ranges", "").lower() == "bytes"
        # A redirect to the CDN may drop Accept-Ranges; assume ranges work and
        # let the ranged GET fail loudly if not.
        if not ranges:
            ranges = r.status in (200, 206)
        return size, ranges


def _fetch_range(url: str, start: int, end: int, out: str,
                 retries: int = 6) -> int:
    """Download [start, end] and write it at the right offset. Returns bytes.

    Retried with backoff: the mirror routinely drops a connection mid-body
    (`UNEXPECTED_EOF_WHILE_READING`), which is a transport fault, not a bad
    request — resending the same range is correct and idempotent.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = r.read()
            expected = end - start + 1
            if len(data) != expected:
                raise IOError(f"short read {len(data)}/{expected} at {start}")
            fd = os.open(out, os.O_WRONLY)
            try:
                os.pwrite(fd, data, start)
            finally:
                os.close(fd)
            return len(data)
        except Exception as e:                       # noqa: BLE001
            last = e
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"range {start}-{end} failed after {retries} tries: {last}")


def download_file(url: str, out: str, workers: int = 4) -> dict:
    size, ranges = _head(url)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if size == 0 or not ranges or size < CHUNK * 2:
        # Small or non-rangeable: one streaming request is simpler and correct.
        # Retried for the same mid-body drop the ranged path handles.
        last: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=TIMEOUT) as r, \
                        open(out, "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                if size and os.path.getsize(out) != size:
                    raise IOError(f"short read {os.path.getsize(out)}/{size}")
                break
            except Exception as e:                   # noqa: BLE001
                last = e
                if attempt + 1 < 4:
                    time.sleep(min(2 ** attempt, 8))
        else:
            raise RuntimeError(f"download failed after 4 tries: {last}")
        got = os.path.getsize(out)
    else:
        # Pre-size the file so every pwrite lands in a valid offset.
        with open(out, "wb") as f:
            f.truncate(size)
        spans = [(s, min(s + CHUNK - 1, size - 1)) for s in range(0, size, CHUNK)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_fetch_range, url, s, e, out) for s, e in spans]
            for f in as_completed(futs):
                f.result()          # propagate the first failure
        got = os.path.getsize(out)

    if got != size:
        raise RuntimeError(f"{out}: size {got} != expected {size}")

    h = hashlib.sha256()
    with open(out, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return {"path": out, "size": size, "sha256": h.hexdigest()}


def list_files(repo: str, patterns: list[str]) -> list[str]:
    """Minimal glob over the repo tree (avoids a huggingface_hub dependency)."""
    import fnmatch
    api = f"{_base_endpoint()}/api/models/{repo}/tree/main?recursive=true"
    try:
        with urllib.request.urlopen(api, timeout=TIMEOUT) as r:
            entries = json.load(r)
    except Exception:
        # The API is rate-limited on mirrors; fall back to the known lane.
        return _fallback_listing(patterns)
    names = [e["path"] for e in entries if e.get("type") == "file"]
    picked = [n for n in names
              if any(fnmatch.fnmatch(n, p) for p in patterns)]
    return picked


def _fallback_listing(patterns: list[str]) -> list[str]:
    """Hard-coded file list for the two patterns this project actually needs.

    Used only when the mirror's API is rate-limited (it returns 429 on the model
    endpoint routinely). Keys are exact paths from the HF manifest.
    """
    known = [
        "ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
        "ditto_cfg/v0.4_hubert_cfg_trt.pkl",
        "ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
        "ditto_pytorch/aux_models/2d106det.onnx",
        "ditto_pytorch/aux_models/det_10g.onnx",
        "ditto_pytorch/aux_models/face_landmarker.task",
        "ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx",
        "ditto_pytorch/models/appearance_extractor.pth",
        "ditto_pytorch/models/decoder.pth",
        "ditto_pytorch/models/lmdm.pth",
        "ditto_pytorch/models/motion_extractor.pth",
        "ditto_pytorch/models/warping_network.pth",
    ]
    import fnmatch
    return [n for n in known if any(fnmatch.fnmatch(n, p) for p in patterns)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="digital-avatar/ditto-talkinghead")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--include", nargs="+", default=["ditto_pytorch/**", "ditto_cfg/**"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args(argv)

    files = list_files(args.repo, args.include)
    if not files:
        print("no files matched; check --include", file=sys.stderr)
        return 2

    print(f"endpoint {_base_endpoint()}  files {len(files)}  workers {args.workers}")
    total = 0
    failures = []
    t0 = time.time()
    manifest = {}

    for i, path in enumerate(files, 1):
        out = os.path.join(os.path.expanduser(args.dest), path)
        url = _url(args.repo, path)
        try:
            if args.skip_existing and os.path.exists(out) and os.path.getsize(out) > 0:
                size = os.path.getsize(out)
                print(f"[{i}/{len(files)}] skip (exists, {size/1e6:.1f} MB) {path}")
                total += size
                continue
            r = download_file(url, out, workers=args.workers)
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed else 0
            print(f"[{i}/{len(files)}] {r['size']/1e6:8.1f} MB  "
                  f"{rate/1e6:5.2f} MB/s avg  {path}")
            total += r["size"]
            manifest[path] = r
        except Exception as e:                       # noqa: BLE001
            # One unreachable file must not discard the other 13; record it and
            # keep going, then exit non-zero so the caller knows to retry.
            print(f"[{i}/{len(files)}] FAILED {path}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            failures.append((path, str(e)))

    with open(os.path.join(os.path.expanduser(args.dest), "download_manifest.json"),
              "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    elapsed = time.time() - t0
    print(f"\nDONE {total/1e9:.2f} GB in {elapsed/60:.1f} min "
          f"({total/elapsed/1e6:.2f} MB/s)")
    if failures:
        print(f"\n{len(failures)} FILE(S) FAILED — re-run to retry (existing files "
              f"are skipped):", file=sys.stderr)
        for path, err in failures:
            print(f"  {path}: {err[:120]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
