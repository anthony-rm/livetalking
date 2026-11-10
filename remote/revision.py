"""Code-revision identity for the host and the remote GPU worker.

Why this exists
---------------
A deploy that only copies files cannot tell you whether the box is really
running what you just pushed. Before this module, ``Health`` reported the GPU
name and the resident backend but nothing about the *code*, so "deployed" was an
act of faith. Every ``Health`` reply now carries a short revision string and the
deploy refuses to report success when it does not match the local one.

Resolution order (first hit wins)
---------------------------------
    1. ``LT_REVISION`` env var          -- set by the deploy script / systemd
    2. ``.deploy/revision`` in the repo -- written next to the code it describes
    3. ``git rev-parse --short HEAD``   -- a plain checkout with .git present
    4. ``""``                           -- unknown (e.g. an rsynced tree, no git)

The value is also dirty-marked: revision ``abc1234`` means "clean tree at
abc1234", ``abc1234+`` means "abc1234 plus uncommitted edits". That is the one
case that must be loud, because the box then runs code that exists in no commit.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

REVISION_FILE = os.path.join(".deploy", "revision")


def _run_git(args, cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd or os.getcwd(),
            capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_revision(cwd: Optional[str] = None) -> str:
    """``<short-sha>`` or ``<short-sha>+`` when the tree has local edits."""
    sha = _run_git(["rev-parse", "--short", "HEAD"], cwd)
    if not sha:
        return ""
    status = _run_git(["status", "--porcelain", "--untracked-files=no"], cwd)
    dirty = status is None or bool(status)
    return f"{sha}+" if dirty else sha


def file_revision(path: str = REVISION_FILE) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def write_revision(revision: str, path: str = REVISION_FILE) -> str:
    """Persist the revision next to the code. Returns the path written."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write((revision or "").strip() + "\n")
    os.replace(tmp, path)
    return path


def current_revision(cwd: Optional[str] = None) -> str:
    """Best-known revision of THIS checkout, in the order documented above."""
    env = (os.environ.get("LT_REVISION") or "").strip()
    if env:
        return env
    on_disk = file_revision()
    if on_disk:
        return on_disk
    return git_revision(cwd)


def matches(remote_revision: str, local_revision: str) -> bool:
    """Compare revisions, ignoring the dirty marker.

    A dirty local tree (``abc1234+``) legitimately differs from what the box
    runs only after a deploy copies those edits, and the box then gets the same
    ``abc1234+``. Ignoring ``+`` keeps the check honest without rejecting a
    deploy that shipped uncommitted work on purpose.
    """
    def norm(value: str) -> str:
        return (value or "").strip().rstrip("+")
    if not norm(remote_revision) or not norm(local_revision):
        return False
    return norm(remote_revision) == norm(local_revision)
