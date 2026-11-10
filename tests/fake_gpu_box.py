"""Volunteer test double: a FAKE GPU box reachable over the real `ssh`/`rsync`.

Why a double
------------
AGENTS.md forbids this agent from connecting to the real GPU box. But a deploy
script that has never been executed is exactly the kind of "it looks right" code
that fails on the box at 2 a.m. So this module builds a throwaway "remote" on
this machine:

* shim `ssh` and `rsync` executables are placed at the FRONT of PATH. `ssh`
  runs the remote command locally with HOME pointed at the fake box; `rsync`
  rewrites `host:path` to a local path and delegates to the real `/usr/bin/rsync`.
  The deploy script's own command line is preserved, so quoting, `~` expansion,
  `ln -sfn`/`mv -T` behaviour and rsync exclude patterns are genuinely exercised.

Limitations, stated honestly: the shimmed rsync has no wire protocol and only
understands the "local dir -> host:dir/" form the deploy uses. That is enough to
catch the failure classes we care about (wrong path, wrong quoting, a missing
exclude, a symlink swap that nests instead of replaces).
"""
from __future__ import annotations

import os
import sys

SSH_SHIM = r"""#!/usr/bin/env bash
# Fake ssh: `ssh [opts] user@host <command>` -> run <command> in $FAKE_GPU_HOME.
# HOME is overridden so a `~` in the remote command resolves to the fake box's
# home, exactly as it would on the real host.
set -u
HOST=""
while [ $# -gt 0 ]; do
  case "$1" in
    -p|-o|-e|-i|-l|-F) shift 2 ;;
    -*) shift ;;
    *) HOST="$1"; shift; break ;;
  esac
done
CMD="${1:-true}"
if [ -z "$HOST" ]; then echo "fake-ssh: no host" >&2; exit 255; fi
cd "$FAKE_GPU_HOME" || exit 255
export HOME="$FAKE_GPU_HOME"
exec /bin/bash -c "$CMD"
"""

RSYNC_SHIM = r'''#!/usr/bin/env python3
"""Fake rsync: `rsync [opts] [-e cmd] SRC/ host:DEST/` -> real rsync, local DEST.

Parsing is deliberate about `-e`/`--rsh` taking a VALUE, which the obvious bash
loop gets wrong (it treats the rsh command as the source). That mistake is what
this file exists to avoid repeating.
"""
import os
import subprocess
import sys

TAKES_VALUE = {"-e", "--rsh", "-f", "--exclude", "--include", "--exclude-from",
               "--include-from", "--rsync-path", "--partial-dir", "--backup-dir",
               "--timeout", "--port"}

args = sys.argv[1:]
passthru = []
positional = []
i = 0
while i < len(args):
    a = args[i]
    if a in TAKES_VALUE:
        # Keep --exclude/--include (and their pattern) for the real rsync; drop
        # only the transport option and its value.
        if a in ("--exclude", "--include", "--exclude-from", "--include-from"):
            passthru.extend([a, args[i + 1]])
        i += 2
        continue
    if a.startswith("-e") and len(a) > 2:
        i += 2                      # "-essh ..." glued form
        continue
    if a.startswith("--rsh="):
        i += 1
        continue
    if a.startswith("-"):
        passthru.append(a)
        i += 1
        continue
    positional.append(a)
    i += 1

if len(positional) < 2:
    sys.stderr.write("fake-rsync: expected SRC and DEST, got %r\n" % (positional,))
    raise SystemExit(1)

src, dest = positional[0], positional[-1]
home = os.environ["FAKE_GPU_HOME"]
if ":" in dest and not dest.startswith("/"):
    dest = dest.split(":", 1)[1]
if dest == "~":
    dest = home
elif dest.startswith("~/"):
    dest = os.path.join(home, dest[2:])
elif not dest.startswith("/"):
    dest = os.path.join(home, dest)

os.makedirs(dest, exist_ok=True)
cmd = ["/usr/bin/rsync"] + passthru + [src, dest]
raise SystemExit(subprocess.call(cmd))
'''


def install(workdir: str) -> dict:
    """Create <workdir>/{home,bin} with shims; return the environment to use."""
    home = os.path.join(workdir, "home")
    bin_dir = os.path.join(workdir, "bin")
    os.makedirs(home, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)
    for name, body in (("ssh", SSH_SHIM), ("rsync", RSYNC_SHIM)):
        path = os.path.join(bin_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(path, 0o755)
    env = dict(os.environ)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["FAKE_GPU_HOME"] = home
    return {"home": home, "bin": bin_dir, "env": env}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m tests.fake_gpu_box <workdir>", file=sys.stderr)
        raise SystemExit(2)
    info = install(sys.argv[1])
    print(info["home"])
