"""Push this working tree to the remote GPU box and restart its worker.

Design
------
* **rsync, not git.** The GPU box has no reliable GitHub access, and you want
  the code that is *on your disk right now* -- including the uncommitted edits
  you are iterating on -- to be what the box runs. Git's own transfer would
  silently ship the last commit and lie about it. ``remote/revision.py`` labels
  such a tree ``<sha>+`` so the difference is never invisible.
* **Release dirs + symlink.** Each deploy lands in
  ``<remote>/releases/<rev>-<stamp>/`` and ``current`` is swapped atomically, so
  a failure mid-copy never leaves a half-written tree in service and rollback is
  one symlink flip. Models, avatars, checkpoints and the conda env live outside
  the release dir and are shared, so a deploy is ~5.7 MB of source, not 40 GB.
* **Verify, don't assume.** After restarting, the deploy asks the worker for its
  ``Health`` and asserts the reported ``revision`` matches the local one and the
  device is ``cuda``. A deploy that cannot prove that fails and rolls back.
* **Refuse to interrupt a session.** ``Health.busy`` is true while a GPU call is
  in flight; the deploy waits for idle unless ``--force``.

Hard rule: this script is for the USER to run. An agent must never execute it
against the box (see AGENTS.md) -- it is written so the user has one command.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import NoReturn, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Settings the user can override with env vars. Defaults match the deployment
# this repo is set up for (SSH credentials already pinned in data/remote_worker.json).
DEFAULT_REMOTE_DIR = os.environ.get("LT_REMOTE_DIR", "~/livetalking")
DEFAULT_PYTHON = os.environ.get("LT_REMOTE_PYTHON", "python")
DEFAULT_PORT = os.environ.get("LT_REMOTE_PORT", "50051")
KEEP_RELEASES = int(os.environ.get("LT_REMOTE_KEEP_RELEASES", "3"))

# Interpreter discovery. A conda env is NOT on PATH for a non-interactive SSH
# command (conda init lives in .bashrc, which non-interactive shells skip), so
# "python" alone resolves to the SYSTEM interpreter on a freshly configured box —
# which has no torch. Discovery therefore runs in two phases:
#
#   1. glob for real environments (<base>/envs/<name>/bin/python) — this catches
#      the layout conda actually uses, including `~/.conda/envs/...` for a
#      user-level install and any prefix the user chose at create time
#   2. then the system interpreters, as a last resort
#
# The first candidate that can `import torch, grpc` wins; the resolved absolute
# path is cached in <remote>/.deploy/python and reused by later restarts.
ENV_GLOBS = [
    '"$HOME"/.conda/envs/*/bin/python',        # user-level conda envs (default
                                              # for `conda create` without a base)
    '"$HOME"/*/envs/*/bin/python',             # miniconda3/anaconda3/mambaforge
    '"$HOME"/*/envs/*/bin/python3',            # ... when only python3 is linked
    "/opt/*/envs/*/bin/python",                # system-wide conda
    "/opt/conda/envs/*/bin/python",
]

# Same locations as concrete shell globs, for expansion on the BOX by the remote
# shell. Kept separate from ENV_GLOBS because a pattern stored in a variable is
# NOT re-globbed when expanded unquoted -- the literal "$HOME" reached the -x
# test, found nothing, and the resolver fell through to bare python3.
ENV_GLOB_SH = [
    '"$HOME"/.conda/envs/*/bin/python',
    '"$HOME"/*/envs/*/bin/python',
    '"$HOME"/*/envs/*/bin/python3',
    "/opt/*/envs/*/bin/python",
    "/opt/conda/envs/*/bin/python",
]

PYTHON_CANDIDATES = [
    '"$HOME"/livetalking/venv/bin/python',
    '"$HOME"/.venv/bin/python',
    "python3",
    "python",
]

# ─── helpers ───────────────────────────────────────────────────────────────
def say(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


def rq(path: str) -> str:
    """Quote a remote path for the remote shell, expanding a leading `~`.

    ``shlex.quote("~/livetalking")`` yields a literal ``~`` that the remote
    shell never expands, so every command would look for a directory named
    ``~``. Expanding it to ``$HOME`` keeps the quoting safe (the remainder is
    quoted separately) and works on any POSIX shell.
    """
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def fail(msg: str) -> "NoReturn":
    print(f"[deploy] FAILED: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


class DeployError(RuntimeError):
    pass


# ─── ssh / credentials (reuse the app's keychain + pinned host key) ────────
def load_target(override: Optional[str] = None) -> dict:
    """Return {username, host, port, password?} for the GPU box.

    Reuses ``remote/settings.py``: username/host from data/remote_worker.json,
    password from the OS keychain. If the keychain entry is missing we fall back
    to plain ``ssh``/``rsync`` with your normal SSH agent -- which is what you
    want anyway when a key is installed.
    """
    from remote import settings
    creds = settings.get_credentials()  # may be None when only a key is used
    username, host = settings.identity()
    if override:
        # "user@host" / "user@host:port" accepted for one-off targets.
        target = override
        if ":" in target and not target.startswith("ssh://"):
            target, _, port = target.partition(":")
        else:
            port = ""
        if "@" in target:
            username, _, host = target.partition("@")
        else:
            host = target
        return {"username": username, "host": host,
                "port": int(port) if port else 22,
                "password": (creds or {}).get("password") or "", "source": "cli"}
    if not host:
        fail("no GPU box configured: set username/host in /admin.html "
             "(Remote GPU Worker card) or pass --target user@host")
    return {"username": username, "host": host, "port": 22,
            "password": (creds or {}).get("password") or "", "source": "settings"}


def _have(binary: str) -> bool:
    from shutil import which
    return which(binary) is not None


def ssh_base(target: dict) -> list:
    user = f"{target['username']}@" if target["username"] else ""
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-p", str(target["port"]), f"{user}{target['host']}"]
    return cmd


RC_MARK = "__LT_RC="


def _split_rc(stdout: str) -> tuple[str, int | None]:
    """Pull the ``__LT_RC=`` marker off the end of a remote script's output.

    ssh exits 255 for a connection/auth failure AND for a remote command that
    happens to exit 255 (or is killed). Those are not the same problem, and the
    deploy used to report the second as "no usable key". The remote command now
    always ends with `echo __LT_RC=$?`, so a real exit code survives ssh's own.
    """
    if not stdout:
        return "", None
    lines = stdout.rstrip("\n").split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(RC_MARK):
            try:
                rc = int(lines[i][len(RC_MARK):].strip())
            except ValueError:
                return stdout, None
            del lines[i]
            return "\n".join(lines), rc
    return stdout, None


def run_ssh(target: dict, remote_cmd: str, *, check: bool = True,
            timeout: float = 120.0, quiet: bool = False
            ) -> subprocess.CompletedProcess:
    """Run one command on the GPU box.

    Password auth is deliberately NOT wired here: sshpass leaks via ``ps`` and
    the whole point of the keychain work was to avoid putting the password on a
    command line. If no key works yet, install one once with ``--setup-key``.

    The command runs inside a subshell with an exit-code marker appended, so a
    remote `exit 4` is reported as 4 rather than arriving as ssh's 255 and being
    blamed on the SSH key.
    """
    cmd = ssh_base(target) + [f"( {remote_cmd} ); echo {RC_MARK}$?"]
    if not quiet:
        say(f"ssh$ {remote_cmd}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        fail(f"ssh timed out after {timeout:g}s running: {remote_cmd}")
    out, rc = _split_rc(proc.stdout)
    proc.stdout = out
    if rc is None:
        # No marker at all: the shell never ran, so this IS a transport failure.
        if proc.returncode == 255:
            fail(ssh_failure_hint(target, proc.stderr))
        proc.returncode = proc.returncode or 1
    else:
        proc.returncode = rc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        fail(f"remote command failed ({proc.returncode}): {remote_cmd}\n{detail}")
    return proc


def ssh_failure_hint(target: dict, stderr: str) -> str:
    detail = (stderr or "").strip()
    return (
        "SSH connection failed -- the command never ran, so this is transport "
        "(key, host key, forwarding), not the deploy.\n"
        f"    target: {target.get('username')}@{target.get('host')}:"
        f"{target.get('port')}\n"
        f"    ssh said: {detail or '(nothing)'}\n"
        "Test it directly:\n"
        f"    ssh -p {target.get('port')} "
        f"{target.get('username')}@{target.get('host')} true\n"
        "If that asks for a password, install the key once:\n"
        "    ./deploy.sh --setup-key")


def run_script(target: dict, script: str, *, label: str = "script",
               timeout: float = 300.0, check: bool = True
               ) -> subprocess.CompletedProcess:
    """Run a multi-line script on the box from a FILE, not on the command line.

    Sending the text as `ssh host '...script...'` puts it in that bash process's
    own argv, so `pgrep -f remote.worker` inside the restart script matched the
    shell running the restart and killed its own session. Writing the script to a
    file first keeps the cmdline at `bash /tmp/...`, which no pattern matches --
    and it stops the deploy log from echoing hundreds of lines of shell.
    """
    import base64
    import uuid
    path = f"/tmp/.lt-deploy-{label}-{uuid.uuid4().hex[:12]}.sh"
    blob = base64.b64encode(script.encode()).decode()
    say(f"script: {label} ({len(script.splitlines())} lines on the box)")
    write = run_ssh(target,
                    f"printf '%s' {blob} | base64 -d > {path} && chmod 700 {path}",
                    timeout=60.0, quiet=True)
    if write.returncode != 0:
        fail(f"could not write {label} script to the box")
    cmd = ssh_base(target) + [f"( bash {path} ); rc=$?; rm -f {path}; "
                              f"echo {RC_MARK}$rc"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        fail(f"{label} timed out after {timeout:g}s on the box")
    out, rc = _split_rc(proc.stdout)
    proc.stdout = out
    if rc is None:
        if proc.returncode == 255:
            fail(ssh_failure_hint(target, proc.stderr))
        proc.returncode = proc.returncode or 1
    else:
        proc.returncode = rc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        fail(f"{label} failed on the box ({proc.returncode})\n{detail}")
    return proc


def public_key() -> str:
    """This machine's SSH public key, or a hard failure with instructions."""
    for name in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"):
        path = os.path.expanduser(f"~/.ssh/{name}")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return fh.read().strip()
    fail("no SSH public key found in ~/.ssh (looked for id_ed25519.pub, "
         "id_rsa.pub, id_ecdsa.pub). Create one with:\n"
         "    ssh-keygen -t ed25519 -C \"$(whoami)@$(hostname)\"")


def print_key(target: dict) -> None:
    """Show the key + the exact paste commands. Never needs a password.

    The manual path matters when the user has no admin rights: the server may
    forbid password auth, `ssh-copy-id` may not exist, and sudo is unavailable.
    Appending one line to ~/.ssh/authorized_keys needs no privileges at all.
    """
    key = public_key()
    say("public key:")
    print(f"\n    {key}\n")
    say(f"paste these on the box as {target['username']} (no sudo needed):")
    print(f"""
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    grep -qxF '{key}' ~/.ssh/authorized_keys 2>/dev/null || \\
      echo '{key}' >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
""")
    say("then verify from here:")
    print(f"    ssh -o BatchMode=yes -p {target['port']} "
          f"{target['username']}@{target['host']} true && echo SSH_OK\n")


def setup_key(target: dict) -> None:
    """Install this machine's public key on the box, once.

    Uses paramiko with the password already in the OS keychain. The password is
    passed in memory (never argv), the host key is compared against the pinned
    fingerprint from `remote/ssh_test.py`, and the remote action is a single
    appended line -- idempotent, no sudo, nothing else is modified.
    """
    import hashlib
    import paramiko
    from remote import settings

    pubkey = public_key()

    password = target.get("password") or ""
    if not password:
        fail("no password in the OS keychain for this host, so the one-time key "
             "install cannot authenticate. Either store the password in "
             "/admin.html -> Remote GPU Worker, or paste this key into the box "
             "yourself:\n"
             f"    # on the box, as {target['username']}:\n"
             f"    mkdir -p ~/.ssh && chmod 700 ~/.ssh\n"
             f"    echo {shlex.quote(pubkey)} >> ~/.ssh/authorized_keys\n"
             "    chmod 600 ~/.ssh/authorized_keys")

    pinned = settings.pinned_host_keys().get(target["host"])
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=target["host"], port=target["port"],
                       username=target["username"], password=password,
                       timeout=15, auth_timeout=15, banner_timeout=15,
                       allow_agent=True, look_for_keys=True)
    except Exception as exc:  # noqa: BLE001
        fail(f"could not log in to {target['host']}: {exc}")
    try:
        transport = client.get_transport()
        if transport is None:
            fail("SSH transport closed before the host key could be checked")
        fp = hashlib.sha256(transport.get_remote_server_key().asbytes()).hexdigest()
        if pinned and pinned != fp:
            fail("HOST KEY CHANGED -- refusing. Possible man-in-the-middle. "
                 f"Expected {pinned[:16]}..., got {fp[:16]}...")
        if not pinned:
            say(f"pinning host key {fp[:16]}... (first contact)")
            settings.pin_host_key(target["host"], fp)

        # Idempotent: only append if the exact key line is not already there.
        cmd = ("umask 077; mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && "
               "grep -qxF \"$LT_KEY\" ~/.ssh/authorized_keys || "
               "printf '%s\\n' \"$LT_KEY\" >> ~/.ssh/authorized_keys")
        stdin, stdout, stderr = client.exec_command(
            f"LT_KEY={shlex.quote(pubkey)} bash -c {shlex.quote(cmd)}", timeout=20)
        err = stderr.read().decode("utf-8", "replace").strip()
        if err:
            fail(f"remote key install reported: {err}")
    finally:
        client.close()

    proc = subprocess.run(ssh_base(target) + ["true"], capture_output=True, text=True)
    if proc.returncode != 0:
        fail("key was written but a key-based login still fails "
             f"({proc.stderr.strip()}). Check ~/.ssh permissions on the box "
             f"({target['username']} home, .ssh 700, authorized_keys 600).")
    say(f"SSH key installed; key-based login to {target['host']} works")


# ─── preflight ─────────────────────────────────────────────────────────────
def local_preflight() -> str:
    """Checks that need no network at all. Returns the local revision.

    Split from the remote probe so `--dry-run` can answer "is this tree
    deployable?" without touching the box (and so a dry run is genuinely dry).
    """
    from remote.revision import current_revision
    local_rev = current_revision()
    if not local_rev:
        fail("could not determine the local revision (not a git checkout and no "
             ".deploy/revision). Run from the repo root.")
    if local_rev.endswith("+"):
        say(f"WARNING: local tree has uncommitted edits -- shipping revision {local_rev}")

    # Stub freshness: a stale generated stub is the classic silent mismatch.
    proto = os.path.join(REPO_ROOT, "remote/proto/inference.proto")
    stub = os.path.join(REPO_ROOT, "remote/proto/inference_pb2.py")
    if os.path.getmtime(proto) > os.path.getmtime(stub) + 1:
        fail("remote/proto/inference.proto is newer than inference_pb2.py -- run "
             "scripts/gen_remote_stubs.sh first")

    if not _have("rsync"):
        fail("rsync is required (macOS ships it). Not found on PATH.")
    if not _have("ssh"):
        fail("ssh is required. Not found on PATH.")
    return local_rev


def preflight(target: dict, remote_dir: str) -> dict:
    local_rev = local_preflight()
    say(f"local revision {local_rev}  ->  {target['username']}@{target['host']}:{remote_dir}")

    # What the box has available. rsync is preferred (incremental), tar is the
    # fallback for hosts where nobody has root to install rsync -- which is the
    # common case on a shared GPU server. Both are pure userspace.
    probe = run_ssh(
        target,
        "set -e; "
        f"cd {rq(remote_dir)} 2>/dev/null || {{ echo NO_DIR; exit 3; }}; "
        "pwd; "
        "command -v rsync >/dev/null && echo HAS_RSYNC || echo NO_RSYNC; "
        "command -v tar   >/dev/null && echo HAS_TAR   || echo NO_TAR; "
        "command -v gzip  >/dev/null && echo HAS_GZIP  || echo NO_GZIP",
        check=False)
    out = (probe.stdout or "").strip()
    if "NO_DIR" in out or probe.returncode == 3:
        fail(f"remote directory {remote_dir} does not exist. Create it once with\n"
             f"    ssh {target['username']}@{target['host']} 'mkdir -p {remote_dir}'")
    if probe.returncode == 255:
        fail("could not connect over SSH. Check that your key is installed:\n"
             f"    ssh -p {target['port']} {target['username']}@{target['host']} true")

    if "HAS_RSYNC" in out:
        method = "rsync"
    elif "HAS_TAR" in out and "HAS_GZIP" in out:
        method = "tar"
        say("rsync is not installed on the box; using tar over ssh (fine without "
            "root, a little slower on large trees)")
    else:
        fail("neither rsync nor tar+gzip is available on the box. tar ships with "
             "every Linux; if it is missing, ask the administrator for it.")
    resolve_python(target, remote_dir)
    return {"revision": local_rev, "probe": out, "method": method}


# ─── transfer ──────────────────────────────────────────────────────────────
def resolve_python(target: dict, remote_dir: str) -> str:
    """Find a usable interpreter on the box and cache it. Returns the command.

    ``$HOME`` is left unexpanded on purpose: the value is embedded in remote
    shell scripts, so it must expand THERE, not here.

    Order: an explicit ``LT_REMOTE_PYTHON`` wins; otherwise the cached path is
    re-validated (fast path -- ``import torch`` costs seconds, so scanning nine
    candidates on every deploy would be painful) and only rescanned when it no
    longer works, e.g. after the env was recreated.
    """
    if os.environ.get("LT_REMOTE_PYTHON"):
        return os.environ["LT_REMOTE_PYTHON"]

    cache = f"{rq(remote_dir)}/.deploy/python"
    candidates = " ".join(PYTHON_CANDIDATES)
    # Each pattern is written INLINE in the for-list so the remote shell expands
    # it. Storing it in a variable first would suppress globbing and hand the
    # literal pattern to the -x test.
    glob_list = " ".join(ENV_GLOB_SH)
    probe = run_ssh(
        target,
        f"CACHED=$(cat {cache} 2>/dev/null || true); "
        # A cached path is only trusted if it still imports torch AND is not the
        # system interpreter when a real env exists. The old code cached whatever
        # it fell back to, so a box that lacked grpc on the first attempt kept a
        # bare `python3` forever -- self-heal that here.
        'if [ -n "$CACHED" ] && "$CACHED" -c "import torch" >/dev/null 2>&1; then '
        '  echo "PICK=$CACHED"; echo "SRC=cache"; exit 0; fi; '
        'if [ -n "$CACHED" ]; then echo "STALE=$CACHED"; fi; '
        # Phase 1: real environments. `torch` is the gate, NOT `grpc`: on a
        # fresh box the env has CUDA torch but the worker deps are not
        # installed yet, and requiring grpc here sends the deploy to the
        # torch-less system python -- a chicken-and-egg the user then has to
        # untangle by hand.
        'ENVS=""; for p in ' + glob_list + '; do [ -x "$p" ] && ENVS="$ENVS $p"; done; '
        'for p in $ENVS; do case "$p" in */livetalking/*) '
        '  if "$p" -c "import torch" >/dev/null 2>&1; then '
        '    echo "PICK=$p"; echo "SRC=env"; exit 0; fi;; esac; done; '
        'for p in $ENVS; do '
        '  if "$p" -c "import torch" >/dev/null 2>&1; then '
        '    echo "PICK=$p"; echo "SRC=env"; exit 0; fi; done; '
        # Phase 2: plain interpreters on PATH. torch alone is enough to be
        # usable; grpc is reported separately so the deploy can say "install
        # the deps" instead of silently picking a different interpreter.
        'for c in ' + candidates + '; do '
        '  if "$c" -c "import torch" >/dev/null 2>&1; then '
        '    echo "PICK=$c"; echo "SRC=path"; exit 0; fi; '
        '  if [ -z "$FALLBACK" ] && "$c" -c "import sys" >/dev/null 2>&1; then '
        '    FALLBACK="$c"; fi; done; '
        'if [ -n "$FALLBACK" ]; then echo "PICK=$FALLBACK"; echo "SRC=fallback"; '
        'else echo "SRC=none"; fi',
        check=False)
    out = probe.stdout or ""
    import re
    pick = re.search(r"^PICK=(.+)$", out, re.M)
    src = re.search(r"^SRC=(\w+)$", out, re.M)
    stale = re.search(r"^STALE=(.+)$", out, re.M)
    source = src.group(1) if src else "none"

    if stale and source != "cache":
        say(f"ignoring stale cached interpreter {stale.group(1).strip()} "
            f"(it cannot import torch)")

    if source == "none" or not pick:
        fail("no usable python found on the box (tried python3, python, and the "
             "usual conda/venv locations). Install Python 3 there, or point the "
             "deploy at the interpreter explicitly:\n"
             "    LT_REMOTE_PYTHON=$HOME/miniconda3/envs/livetalking/bin/python "
             "./deploy.sh")

    chosen = pick.group(1).strip()
    if source == "cache":
        say(f"remote interpreter: {chosen} (cached)")
    elif source == "fallback":
        say(f"remote interpreter: {chosen}")
        say("WARNING: that interpreter cannot import torch, so it is almost "
            "certainly the system python and NOT your conda env. Nothing was "
            "cached; fix the env, then re-run. To see what is there:")
        say("    ls -d ~/.conda/envs/*/bin/python ~/*/envs/*/bin/python 2>/dev/null")
        say("If your env is elsewhere, name it explicitly:")
        say("    LT_REMOTE_PYTHON=/abs/path/to/envs/livetalking/bin/python ./deploy.sh")
    else:
        say(f"remote interpreter: {chosen}")

    # grpc is a separate question: the worker cannot start without it, but its
    # absence must not change WHICH interpreter is chosen. Report it, and let
    # the deploy continue -- the first run exists precisely to put the source on
    # the box so `pip install -r requirements.txt` can run in this env.
    if source != "fallback":
        check = run_ssh(target,
                        f"{shlex.quote(chosen)} -c 'import grpc' 2>/dev/null "
                        f"&& echo GRPC_OK || echo NO_GRPC", check=False)
        if "NO_GRPC" in (check.stdout or ""):
            target["missing_grpc"] = True
            say("NOTE: grpc is not installed in that env yet, so the worker "
                "cannot start. Install the minimal set (on the box):")
            say(f"    {chosen} -m pip install grpcio opencv-python-headless tqdm")
            say("Then re-run ./deploy.sh --restart-only. No re-copy needed.")

    # Only cache a real choice. Caching the fallback (a bare python3) is what
    # made a later deploy keep using the wrong interpreter with no explanation.
    if source in ("env", "path"):
        run_ssh(target, f"mkdir -p {rq(remote_dir)}/.deploy && "
                        f"printf '%s\\n' {shlex.quote(chosen)} > {cache}",
                check=False)
    return chosen


def _rsync_files(target: dict, release_path: str) -> None:
    """rsync exactly the files ``should_ship`` approves.

    Not `rsync -a src/ dst/` with --exclude patterns: the two transports must
    agree on the file set, and the predicate is the single definition.

    The file list is written to a temp FILE, not piped as "--files-from=-",
    because openrsync (macOS) and GNU rsync disagree about how the sender
    validates relative paths from a stream: the "-" form aborts with
    "invalid path from sender" on dot-directories like .github/. Use a file plus
    --from0, which both implementations read identically.
    """
    files = _collect_files()
    user = f"{target['username']}@" if target["username"] else ""
    dest = f"{user}{target['host']}:{release_path}"
    say(f"rsync {len(files)} files")

    import tempfile
    with tempfile.NamedTemporaryFile("wb", suffix=".files", delete=False) as fh:
        fh.write(b"".join(os.path.relpath(f, REPO_ROOT).encode() + b"\0"
                          for f in files))
        list_path = fh.name
    try:
        proc = subprocess.run(
            ["rsync", "-az", f"--files-from={list_path}", "--from0",
             "-e", f"ssh -o BatchMode=yes -o ConnectTimeout=10 -p {target['port']}",
             f"{REPO_ROOT}/", f"{dest}/"])
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass
    if proc.returncode == 7:
        fail("rsync could not connect (exit 7). Check SSH key access:\n"
             f"    ssh -p {target['port']} {user}{target['host']} true")
    if proc.returncode != 0:
        fail(f"rsync failed with exit {proc.returncode}")


def push_tree(target: dict, remote_dir: str, release_path: str, *,
              dry_run: bool = False, method: str = "rsync") -> None:
    """Copy the working tree into a fresh release dir on the box.

    Two transports, both pure userspace so no root is needed anywhere:
    ``rsync`` when the box has it (incremental, and the default), otherwise
    ``tar | gzip`` streamed over the existing SSH connection. The release dir is
    new every time, so incremental transfer buys little -- tar is a fine
    fallback, just less efficient on a big tree.
    """
    if method == "tar":
        return _push_tar(target, release_path, dry_run=dry_run)

    if dry_run:
        say(f"dry run: would rsync {len(_collect_files())} files")
        return
    _rsync_files(target, release_path)


# Directories excluded wholesale, matched on the repo-relative PATH with an
# exact prefix -- never by basename. `./models` must not also match
# `avatars/musetalk/models/`, which holds vendored .py sources the worker needs.
EXCLUDED_PREFIXES = (
    "models", "data", "gfpgan/weights", ".git", ".venv", "venv",
    "__pycache__", ".deploy", ".pytest_cache", ".mypy_cache", "node_modules",
    "trial", "tmp",
)
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".log", ".mp4", ".mov", ".DS_Store")
EXCLUDED_NAMES = (".env", ".env.local", "livetalking.pid")


def should_ship(relpath: str) -> bool:
    """Whether a repo-relative path belongs in a deploy.

    Deliberately a Python predicate rather than `tar --exclude` patterns: the
    archive is BUILT on macOS (bsdtar) and EXTRACTED on Linux (GNU tar), and the
    two disagree about what `--exclude ./models` means. bsdtar matches it against
    `avatars/musetalk/models/` as well and silently drops the vendored sources,
    which breaks the worker with ModuleNotFoundError only after it is deployed.
    """
    rel = relpath
    while rel.startswith("./"):
        rel = rel[2:]
    rel = rel.strip("/")
    if not rel:
        return False
    parts = rel.split("/")
    if parts[0] in EXCLUDED_PREFIXES:
        return False
    if any(part == "__pycache__" for part in parts):
        return False
    if parts[-1] in EXCLUDED_NAMES:
        return False
    if rel.endswith(EXCLUDED_SUFFIXES):
        return False
    # gfpgan/weights is nested, so check the two-part prefix explicitly.
    return "/".join(parts[:2]) != "gfpgan/weights"


def _push_tar(target: dict, release_path: str, *, dry_run: bool = False) -> None:
    """Stream the tree as a gzipped tarball through ssh.

    The archive is produced in-process with ``tarfile`` so exclusions are decided
    by ``should_ship()`` and not by whichever ``tar`` happens to be installed.
    Bytes go straight into the ssh process, so nothing large lands on disk.
    """
    user = f"{target['username']}@" if target["username"] else ""
    remote = f"mkdir -p {rq(release_path)} && tar -xzf - -C {rq(release_path)}"
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-p", str(target["port"]), f"{user}{target['host']}", remote]

    included = _collect_files()
    size_mb = sum(os.path.getsize(p) for p in included) / 1e6
    say(f"tar over ssh: {len(included)} files, {size_mb:.1f} MB uncompressed")
    if dry_run:
        say("dry run: would stream those files over ssh")
        return

    try:
        extractor = subprocess.Popen(ssh_cmd, stdin=subprocess.PIPE)
    except OSError as exc:
        fail(f"could not start ssh for the tar stream: {exc}")

    import tarfile
    try:
        with tarfile.open(fileobj=extractor.stdin, mode="w|gz") as archive:
            for path in included:
                arcname = os.path.relpath(path, REPO_ROOT)
                archive.add(path, arcname=arcname, recursive=False)
    except (BrokenPipeError, OSError) as exc:
        # The far end died (auth, disk full, no tar). Its exit code is more
        # informative than the broken pipe, so read that below.
        say(f"tar stream interrupted: {type(exc).__name__}")
    finally:
        if extractor.stdin is not None:
            try:
                extractor.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        extractor.wait()
    if extractor.returncode != 0:
        fail(f"remote extract failed (exit {extractor.returncode}); check that "
             f"tar+gzip exist on the box and that {release_path} is writable")


def _collect_files() -> list:
    """Every file under the repo root that should ship, in stable order."""
    collected = []
    for root, dirs, files in os.walk(REPO_ROOT):
        rel_root = os.path.relpath(root, REPO_ROOT)
        if rel_root == ".":
            rel_root = ""
        # Prune excluded directories before descending: walking .git or a 40 GB
        # models/ tree just to discard it would dominate the deploy time.
        dirs[:] = sorted(d for d in dirs
                         if should_ship(os.path.join(rel_root, d) if rel_root else d))
        for name in sorted(files):
            rel = os.path.join(rel_root, name) if rel_root else name
            full = os.path.join(root, name)
            if os.path.islink(full):
                continue          # symlinks are recreated explicitly on the box
            if should_ship(rel):
                collected.append(full)
    return collected


# ─── shared state outside the release dirs ─────────────────────────────────
# Weights and avatar artifacts are far too big to re-copy per deploy, and they
# are not code. They live ONCE at the install root and every release symlinks
# back to them, so a deploy is ~6 MB of source and a rollback cannot lose a
# 3 GB checkpoint.
SHARED_DIRS = ["models", "data", "gfpgan/weights"]


def link_shared(target: dict, remote_dir: str, release_name: str) -> None:
    """Symlink weights/avatars from the install root into the new release."""
    lines = ["set -e", f"BASE={rq(remote_dir)}",
             f"REL=\"$BASE/releases/{release_name}\"",
             'mkdir -p "$BASE/models" "$BASE/data" "$BASE/gfpgan/weights"']
    for rel in SHARED_DIRS:
        parent = os.path.dirname(rel)
        depth = "../../../" if parent else "../../"
        lines.append(f'mkdir -p "$REL/{parent}"' if parent else "")
        lines.append(f'ln -sfn "{depth}{rel}" "$REL/{rel}"')
    # A local .env, if the box has one, must survive deploys too.
    lines.append('if [ -f "$BASE/.env" ]; then ln -sfn "../../.env" "$REL/.env"; fi')
    script = "\n".join(line for line in lines if line.strip()) + "\n"
    run_script(target, script, label="link")


# ─── activation / restart ──────────────────────────────────────────────────
def swap_current(target: dict, remote_dir: str, release_name: str) -> str:
    """Atomically repoint <remote>/current at a release. Returns the new link."""
    script = f"""
set -e
cd {rq(remote_dir)}
mkdir -p releases logs
ln -sfn {shlex.quote("releases/" + release_name)} current.new
# mv -T treats the destination as a file even when it is a symlink to a dir, so
# this REPLACES `current` instead of nesting the new link inside it. mv -T is
# GNU-only; the fallback keeps it portable.
mv -Tf current.new current 2>/dev/null || {{
  rm -rf current.old; mv current current.old 2>/dev/null || true
  mv current.new current
}}
readlink current
"""
    return (run_script(target, script, label="swap", timeout=60).stdout or "").strip()


def activate(target: dict, remote_dir: str, release_name: str, *,
             force: bool = False) -> None:
    """Point <remote>/current at a release and restart the worker."""
    link = swap_current(target, remote_dir, release_name)
    say(f"current -> {link or release_name}")
    if target.get("missing_grpc"):
        say("worker NOT restarted: its dependencies are missing in that "
            "interpreter, so it would die on import. Install them (commands "
            "above), then run: ./deploy.sh --restart-only")
        return
    restart_worker(target, remote_dir, force=force)


def restart_worker(target: dict, remote_dir: str, *, force: bool = False) -> None:
    """Restart the worker process on the box.

    Three strategies, in order of preference:
      1. systemd user or system unit named livetalking-worker (if present)
      2. a pidfile managed by this script (<remote>/.deploy/worker.pid)
      3. a fresh nohup launch with a logfile
    The script never assumes sudo; it tries the unprivileged paths first.
    """
    wait_for_idle(target, remote_dir, force=force)
    py = resolve_python(target, remote_dir)
    port = DEFAULT_PORT
    script = f"""
set -e
BASE={rq(remote_dir)}
PIDF="$BASE/.deploy/worker.pid"
LOG="$BASE/logs/worker.log"
mkdir -p "$BASE/logs" "$BASE/.deploy"

restart_systemd() {{
  if systemctl --user list-unit-files 2>/dev/null | grep -q '^livetalking-worker'; then
    systemctl --user restart livetalking-worker && echo RESTARTED=systemd-user && return 0
  fi
  if systemctl list-unit-files 2>/dev/null | grep -q '^livetalking-worker'; then
    systemctl restart livetalking-worker && echo RESTARTED=systemd-system && return 0
  fi
  return 1
}}

if restart_systemd; then exit 0; fi

# The worker is only importable from the release directory.
cd "$BASE/current" || exit 6

# Preflight: prove this interpreter can actually run the worker BEFORE
# daemonising it. `RESTARTED=DEAD` plus a log tail is a terrible error message
# when the real answer is a missing dependency.
#
# The check is a self-contained `-c` snippet on purpose. It must NOT import a
# project module: `--restart-only` starts whatever release is already at
# current/, which may predate any helper we just wrote locally. `remote.worker
# --check` cannot be used either -- worker.py imports grpc at module scope, so
# it dies with a bare ImportError before its own argument parsing runs.
# Keep the list below in sync with remote/preflight.py (REQUIRED).
PREFLIGHT_PY='
missing = []
# "google.protobuf" matters as much as "grpc": grpcio >= 1.8x does NOT bundle
# protobuf, and the generated inference_pb2 imports it directly. Checking only
# grpc let a worker pass preflight and die on import.
for mod in ("grpc", "google.protobuf", "numpy", "cv2", "tqdm", "torch"):
    try:
        __import__(mod)
    except Exception as exc:
        missing.append("%s: %s" % (mod, exc))
if missing:
    print("missing dependencies (this interpreter):")
    for line in missing:
        print("  " + line)
    raise SystemExit(2)
import torch
print("device: " + ("cuda" if torch.cuda.is_available() else "cpu")
      + (", gpu: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))
'
if ! OUT=$({py} -c "$PREFLIGHT_PY" 2>&1); then
  echo "PREFLIGHT=FAILED"
  printf '%s\n' "$OUT"
  exit 5
fi
echo "PREFLIGHT=OK $OUT"

# pidfile path: stop whatever we started before
if [ -f "$PIDF" ]; then
  OLD=$(cat "$PIDF" 2>/dev/null || true)
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    kill -TERM "$OLD" 2>/dev/null || true
    for i in $(seq 1 40); do kill -0 "$OLD" 2>/dev/null || break; sleep 0.25; done
    kill -0 "$OLD" 2>/dev/null && kill -KILL "$OLD" 2>/dev/null || true
  fi
  rm -f "$PIDF"
fi

# Also catch a worker started by hand (an earlier session, or the foreground
# command in docs/remote-gpu.md). A second worker on the same port would take
# over the GPU silently and answer Health with the OLD revision -- the exact
# failure the post-deploy check exists to catch, so failing to stop it here
# would make that check meaningless.
#
# MATCH-SAFE: -f matches the whole cmdline, and patterns are regexes, so this
# must (a) exclude our own pid and parent chain and (b) not match its own text.
# It runs from a script file now, so our cmdline is "bash /tmp/...", but the
# self-exclusion stays: a hand-run `bash -c` copy would otherwise kill itself.
WORKER_RE='remote[.]worker'
OURS="$$ $PPID"
for pid in $(pgrep -f "$WORKER_RE" 2>/dev/null || true); do
  case " $OURS " in *" $pid "*) continue;; esac
  # Skip anything that is our own shell or an ancestor.
  ppid=$pid
  for _ in 1 2 3 4 5; do
    ppid=$(ps -o ppid= -p "$ppid" 2>/dev/null | tr -d ' ')
    [ -z "$ppid" ] && break
    [ "$ppid" = "$PPID" ] && {{ pid=""; break; }}
  done
  [ -n "$pid" ] || continue
  kill -TERM "$pid" 2>/dev/null || true
done
for i in $(seq 1 40); do
  pgrep -f "$WORKER_RE" >/dev/null 2>&1 || break
  sleep 0.25
done
for pid in $(pgrep -f "$WORKER_RE" 2>/dev/null || true); do
  case " $OURS " in *" $pid "*) continue;; esac
  kill -KILL "$pid" 2>/dev/null || true
done

# Mark where this run starts in the log: it is append-only, so a bare
# `tail -30` can show output from a PREVIOUS worker and misdiagnose the
# current failure.
touch "$LOG" 2>/dev/null || true
LOG_MARK=$(wc -c < "$LOG" 2>/dev/null || echo 0)
nohup {py} -m remote.worker --host 127.0.0.1 --port {port} >> "$LOG" 2>&1 &
echo $! > "$PIDF"
# A worker that dies immediately (bad env, import error) would otherwise show
# up only as "no revision" 90 seconds later; surface it now.
for _i in $(seq 1 20); do
  kill -0 "$(cat "$PIDF")" 2>/dev/null || break
  sleep 0.5
done
if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "RESTARTED=DEAD"
  echo "--- worker log (this run only) ---"
  tail -c +$((LOG_MARK + 1)) "$LOG" 2>/dev/null | tail -n 40 || true
  echo "--- end of log ---"
  exit 4
fi
echo RESTARTED=nohup PID=$(cat "$PIDF")
"""
    proc = run_script(target, script, label="restart", timeout=180, check=False)
    out = proc.stdout or ""
    if "PREFLIGHT=FAILED" in out or proc.returncode == 5:
        fail("the worker CANNOT start with that interpreter -- the dependency "
             "preflight failed, so no process was started and nothing was left "
             "running.\n"
             f"{out.strip()}\n\n"
             "Install the missing modules in TARGET, then re-run:\n"
             f"    {py} -m pip install grpcio opencv-python-headless tqdm\n"
             "Do NOT install -r requirements.txt just to get the worker "
             "listening: it is large, resolves slowly, and is not needed for "
             "the RPC handshake.")
    if proc.returncode == 6:
        fail("~/livetalking/current is missing or broken on the box. It should "
             "be a symlink to a release directory; re-run the deploy.")
    if proc.returncode == 4:
        # The remote script already captured this run's log slice into stdout;
        # print it, otherwise "see the excerpt above" refers to nothing.
        detail = "\n".join(
            line for line in (out or "").splitlines()
            if not line.startswith("PREFLIGHT=OK"))
        fail("the worker started but exited immediately. Nothing is running "
             "now.\n"
             f"{detail.strip() or '(the worker produced no output at all)'}\n\n"
             "That traceback is from THIS run only (earlier runs are excluded "
             "because the log is append-only).")
    if proc.returncode != 0:
        fail(f"worker restart failed ({proc.returncode}): "
             f"{(proc.stderr or proc.stdout or '').strip()}")
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESTARTED="):
            say(f"worker {line.strip()}")
    say("worker restart issued")


def _probe(address: str, timeout: float = 5.0) -> dict:
    """Indirection so the deploy logic is testable without grpc installed."""
    from remote.client import probe
    return probe(address, timeout=timeout)


def wait_for_idle(target: dict, remote_dir: str, *, force: bool = False,
                  timeout: float = 120.0) -> None:
    """Poll Health until the worker is idle; refuse to kill a live session."""
    address = health_address(target)
    if not address:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            info = _probe(address, timeout=5.0)
        except Exception:  # noqa: BLE001 -- unreachable is fine, nothing to kill
            return
        if not info.get("busy"):
            return
        if force:
            say("worker busy -- --force given, restarting anyway")
            return
        say("worker is mid-inference/enhance (busy); waiting...")
        time.sleep(3.0)
    if not force:
        fail("worker stayed busy for %gs. Stop the live session, or re-run with "
             "--force to interrupt it." % timeout)


def health_address(target: dict) -> str:
    """The gRPC host:port to dial from this machine.

    Goes through ``settings.effective_address()`` so SSH-tunnel mode, the
    admin-console override and CLI/env all stay in one place. Falls back to the
    SSH host + worker port only when nothing is configured.
    """
    from remote.settings import effective_address
    address = effective_address()
    if address:
        return address
    host = target.get("host") or ""
    return f"{host}:{DEFAULT_PORT}" if host else ""


# ─── verification ──────────────────────────────────────────────────────────
def verify(target: dict, local_rev: str, *, expect_cuda: bool = True,
           deadline: float = 90.0, remote_dir: str = DEFAULT_REMOTE_DIR) -> dict:
    """Prove the box now serves `local_rev` on CUDA. Raises DeployError."""
    address = health_address(target)
    if not address:
        raise DeployError("cannot verify: no worker address (set it in /admin.html)")

    # In SSH-only mode the worker binds 127.0.0.1 on the box, so verification
    # must go through the tunnel -- otherwise it would dial a port that is
    # correctly not exposed and report a false failure.
    from remote import settings
    if settings.tunnel_enabled():
        from remote import tunnel
        say("opening SSH tunnel for verification")
        if not tunnel.ensure():
            raise DeployError(
                "could not open the SSH tunnel to the worker. Check key access:\n"
                f"    ssh -N -L 127.0.0.1:50051:127.0.0.1:50051 "
                f"-p {target['port']} {target['username']}@{target['host']}\n"
                "and see data/tunnel.log")
        address = tunnel.local_address()

    say(f"verifying worker at {address} ...")
    end = time.time() + deadline
    last_err = ""
    info = {}
    while time.time() < end:
        try:
            info = _probe(address, timeout=5.0)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0)
    else:
        raise DeployError(f"worker did not answer Health within {deadline:g}s: {last_err}")

    remote_rev = info.get("revision") or ""
    if not remote_rev:
        raise DeployError(
            "worker answered but reports no revision -- it is running OLD code "
            "without the revision field. Restart it once by hand, then redeploy.")
    from remote.revision import matches
    if not matches(remote_rev, local_rev):
        raise DeployError(
            f"revision mismatch: box serves {remote_rev!r}, local is {local_rev!r}. "
            "The files copied but the process did not restart, or it restarted "
            "from a different directory.")
    if expect_cuda and info.get("device") != "cuda":
        raise DeployError(
            f"worker is on device {info.get('device')!r}, not 'cuda' -- check the "
            "CUDA torch install in the worker's env")
    say(f"OK: revision {remote_rev} on {info.get('device')} ({info.get('detail') or 'GPU'})")
    try:
        report_weights(target, remote_dir)
    except Exception as exc:  # noqa: BLE001
        # The deploy itself succeeded; a survey failure (odd target dict, ssh
        # hiccup) must not turn a working deploy into a red one.
        say(f"weights: could not survey the box ({type(exc).__name__}: {exc})")
    return info


# What the worker needs on disk to actually serve, grouped by feature. A deploy
# is "files copied + worker restarted"; it is NOT "the box can run", and saying
# so unconditionally is how a broken box looks green. Report, don't fail: a
# weights gap is fixed with scripts/fetch_worker_weights.sh, not a rollback.
WEIGHT_FILES = [
    # (path relative to the install root, feature, min_valid_bytes)
    # config files are legitimately tiny; only real weights need the size gate.
    ("models/musetalkV15/unet.pth", "musetalk inference", 1024 * 1024),
    ("models/musetalkV15/musetalk.json", "musetalk inference", 1),
    ("models/sd-vae/config.json", "musetalk inference", 1),
    ("models/sd-vae/diffusion_pytorch_model.bin", "musetalk inference", 1024 * 1024),
    ("models/wav2lip.pth", "wav2lip inference", 1024 * 1024),
    ("models/face-parse-bisent/79999_iter.pth", "musetalk avatar creation", 1024 * 1024),
    ("models/face-parse-bisent/resnet18-5c106cde.pth", "musetalk avatar creation", 1024 * 1024),
    ("models/dwpose/dw-ll_ucoco_384.pth", "musetalk avatar creation", 1024 * 1024),
    ("models/GFPGANv1.4.pth", "video enhance", 1024 * 1024),
    ("models/RealESRGAN_x4plus.pth", "video enhance", 1024 * 1024),
    ("gfpgan/weights/detection_Resnet50_Final.pth", "video enhance", 1024 * 1024),
    ("gfpgan/weights/parsing_parsenet.pth", "video enhance", 1024 * 1024),
]
# A partially-downloaded weight is worse than a missing one: it loads and then
# fails at inference. anything under its per-file minimum is treated as partial.


def report_weights(target: dict, remote_dir: str) -> None:
    """Print which weight files exist on the box, grouped by feature."""
    checks = " ".join(shlex.quote(rel) for rel, _f, _m in WEIGHT_FILES)
    script = f"""
BASE={rq(remote_dir)}
for rel in {checks}; do
  p="$BASE/$rel"
  if [ -f "$p" ]; then
    sz=$(stat -c%s "$p" 2>/dev/null || echo 0)
    echo "HAVE $sz $rel"
  else
    echo "MISS 0 $rel"
  fi
done
"""
    proc = run_script(target, script, label="weights", timeout=120, check=False)
    out = proc.stdout or ""
    mins = {rel: minb for rel, _f, minb in WEIGHT_FILES}
    have, miss, partial = [], {}, []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3 or parts[0] not in ("HAVE", "MISS"):
            continue
        state, size, rel = parts
        if state == "MISS":
            miss[rel] = True
        elif int(size) < mins.get(rel, 0):
            partial.append(rel)
        else:
            have.append(rel)

    if not miss and not partial:
        say(f"weights: all {len(have)} files present")
        return

    missing_features = sorted({feat for rel, feat, _m in WEIGHT_FILES
                               if rel in miss or rel in partial})
    print()
    say("WARNING: the worker is running, but these weights are missing on the "
        f"box, so {', '.join(missing_features)} will FAIL until they are fetched:")
    for rel in sorted(miss):
        print(f"    missing   {rel}")
    for rel in sorted(partial):
        print(f"    incomplete {rel}")
    print("    fix on the box:  bash scripts/fetch_worker_weights.sh")
    print("    (or copy them to ~/livetalking/models/)")


# ─── rollback ──────────────────────────────────────────────────────────────
def list_releases(target: dict, remote_dir: str) -> list:
    out = run_ssh(target, f"ls -1t {rq(remote_dir)}/releases 2>/dev/null || true")
    return [line for line in (out.stdout or "").splitlines() if line.strip()]


def release_revision(target: dict, remote_dir: str, release_name: str) -> str:
    """Read the revision a release was stamped with.

    Parsing the directory name is wrong: the stamp itself contains dashes, so a
    revision like ``c32c4a3+`` followed by ``20261109-143602`` does not split
    cleanly on the first or last dash. The ``.deploy/revision`` file written at
    deploy time is the authoritative record.
    """
    path = f"{rq(remote_dir)}/releases/{shlex.quote(release_name)}/.deploy/revision"
    out = run_ssh(target, f"cat {path} 2>/dev/null || true", check=False)
    return (out.stdout or "").strip()


def rollback(target: dict, remote_dir: str, steps: int = 1, *,
             force: bool = False) -> None:
    releases = list_releases(target, remote_dir)
    if len(releases) <= steps:
        fail(f"only {len(releases)} release(s) on the box; cannot go back {steps}")
    target_release = releases[steps]
    say(f"rolling back to {target_release}")
    activate(target, remote_dir, target_release, force=force)
    # Verify against the revision that release was stamped with, so a rollback
    # proves the OLD code is actually serving -- not merely that some worker
    # answered.
    expected_rev = release_revision(target, remote_dir, target_release)
    if not expected_rev:
        fail(f"release {target_release} has no .deploy/revision stamp, so the "
             f"rollback cannot be verified. Check it on the box, or roll back to "
             f"a release created by ./deploy.sh")
    info = verify(target, expected_rev, expect_cuda=True, remote_dir=remote_dir)
    say(f"rollback verified: {info.get('revision')}")


def prune(target: dict, remote_dir: str, keep: int) -> None:
    if keep < 1:
        return
    script = (f"cd {rq(remote_dir)}/releases && "
              f"ls -1t | tail -n +{keep + 1} | xargs -r rm -rf")
    run_ssh(target, script, check=False)


# ─── cleanup ───────────────────────────────────────────────────────────────
def download_avatars(target: dict, remote_dir: str, avatar_ids: list) -> None:
    """Copy avatar artifacts (gitignored, huge) up to the box on demand.

    They go to the SHARED ``<remote>/data/avatars`` -- not into a release dir --
    because every release symlinks ``data`` back there, so one copy serves all
    of them and survives a rollback.
    """
    user = f"{target['username']}@" if target["username"] else ""
    for avatar_id in avatar_ids:
        src = os.path.join(REPO_ROOT, "data/avatars", avatar_id)
        if not os.path.isdir(src):
            fail(f"no such avatar: {src}")
        run_ssh(target, f"mkdir -p {rq(remote_dir)}/data/avatars")
        dest = (f"{user}{target['host']}:{remote_dir}/data/avatars/{avatar_id}/")
        cmd = ["rsync", "-az", "--stats",
               "-e", f"ssh -o BatchMode=yes -p {target['port']}", src + "/", dest]
        say(f"shipping avatar {avatar_id}")
        proc = subprocess.run(cmd)
        if proc.returncode == 7:
            fail("rsync could not connect (exit 7); check SSH key access")
        if proc.returncode != 0:
            fail(f"avatar sync failed for {avatar_id} (exit {proc.returncode})")


# ─── CLI ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deploy this working tree to the remote GPU worker, restart "
                    "it, and verify the revision it serves.")
    p.add_argument("--target", default=os.environ.get("LT_REMOTE_TARGET", ""),
                   help="user@host[:port]; default from data/remote_worker.json")
    p.add_argument("--dir", default=DEFAULT_REMOTE_DIR,
                   help=f"remote install root (default {DEFAULT_REMOTE_DIR})")
    p.add_argument("--dry-run", action="store_true",
                   help="preflight + show the rsync plan; change nothing")
    p.add_argument("--yes", "-y", action="store_true",
                   help="skip the confirmation prompt")
    p.add_argument("--force", action="store_true",
                   help="restart even if the worker reports a live (busy) session")
    p.add_argument("--no-restart", action="store_true",
                   help="copy files but do not restart the worker")
    p.add_argument("--restart-only", action="store_true",
                   help="skip the copy: just restart the worker on the version "
                        "already at ~/livetalking/current (use after installing "
                        "dependencies by hand)")
    p.add_argument("--rollback", nargs="?", const=1, type=int, default=None,
                   metavar="N", help="activate the Nth previous release (default 1)")
    p.add_argument("--avatars", nargs="*", default=[],
                   help="also ship these data/avatars/<id> directories")
    p.add_argument("--setup-key", action="store_true",
                   help="one-time: install this machine's public key on the box "
                        "(uses the keychain password over paramiko, never argv)")
    p.add_argument("--print-key", action="store_true",
                   help="print this machine's public key and the exact commands "
                        "to paste it on the box (no password needed)")
    p.add_argument("--keep", type=int, default=KEEP_RELEASES,
                   help=f"releases to keep on the box (default {KEEP_RELEASES})")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    remote_dir = args.dir.rstrip("/")
    try:
        target = load_target(args.target or None)
    except DeployError as exc:
        fail(str(exc))

    say(f"target: {target['username']}@{target['host']} ({target['source']} credentials)")

    if args.setup_key:
        setup_key(target)
        return 0
    if args.print_key:
        print_key(target)
        return 0

    if args.rollback is not None:
        rollback(target, remote_dir, args.rollback, force=args.force)
        return 0

    if args.restart_only:
        # No copy, no symlink swap: the point is to start a worker on code that
        # is ALREADY at current/ (typically after `pip install` by hand).
        say("restart-only: not copying any files")
        restart_worker(target, remote_dir, force=args.force)
        # `current` is a symlink, so read the revision through it rather than
        # under releases/ (release_revision resolves names inside that dir).
        rev = (run_ssh(target, f"cat {rq(remote_dir)}/current/.deploy/revision "
                               f"2>/dev/null || true", check=False).stdout or "").strip()
        info = verify(target, rev, expect_cuda=True, remote_dir=remote_dir)
        say(f"verified: revision {info.get('revision')} on {info.get('device')}")
        return 0

    if args.dry_run:
        # Strictly local: no SSH, no rsync. Safe to run anywhere, any time, and
        # it cannot be mistaken for a deploy that half-happened.
        local_rev = local_preflight()
        say(f"dry run: local preflight passed (revision {local_rev})")
        say(f"would deploy to {target['host']}:{remote_dir}/releases/"
            f"{local_rev}-<stamp>/ then restart the worker")
        files = _collect_files()
        total = sum(os.path.getsize(f) for f in files)
        say(f"would transfer {len(files)} files ({total / 1e6:.1f} MB): "
            f"source only — no models/, data/, .env, .git or caches")
        say("nothing was sent and nothing was changed")
        return 0

    state = preflight(target, remote_dir)
    local_rev = state["revision"]
    say(f"transfer method: {state.get('method', 'rsync')}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    release_name = f"{local_rev}-{stamp}"
    release_path = f"{remote_dir}/releases/{release_name}"

    if not args.yes:
        print(f"\nAbout to deploy {local_rev} -> {target['host']}:{release_path}")
        print("This copies source files (no models/data) and restarts the worker.")
        reply = input("Continue? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            say("aborted")
            return 1

    run_ssh(target, f"mkdir -p {rq(remote_dir)}/releases "
                    f"{rq(remote_dir)}/logs {rq(remote_dir)}/.deploy")
    try:
        push_tree(target, remote_dir, release_path, method=state.get("method", "rsync"))
        # Stamp the revision next to the code it describes, so `Health` reports
        # it even on a box with no git history.
        run_ssh(target, f"mkdir -p {rq(release_path)}/.deploy && "
                        f"printf '%s\\n' {shlex.quote(local_rev)} > "
                        f"{rq(release_path)}/.deploy/revision")
        link_shared(target, remote_dir, release_name)
        if args.no_restart:
            # Files are in place and `current` points at them, but the running
            # worker still has the OLD code resident -- swapping the symlink
            # does not reload a process. Say so instead of implying success.
            link = swap_current(target, remote_dir, release_name)
            say(f"current -> {link or release_name}; worker NOT restarted, so it "
                f"still serves the previous code until you restart it")
            return 0
        activate(target, remote_dir, release_name, force=args.force)
        verify(target, local_rev, remote_dir=remote_dir)
    except DeployError as exc:
        fail(str(exc))
    except SystemExit:
        raise

    if args.avatars:
        download_avatars(target, remote_dir, args.avatars)

    prune(target, remote_dir, args.keep)
    say(f"deployed {local_rev}. Roll back with: ./deploy.sh --rollback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
