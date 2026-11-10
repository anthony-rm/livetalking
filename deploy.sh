#!/usr/bin/env bash
# Push local edits to the remote GPU worker and restart it.
#
#   ./deploy.sh --dry-run          # preflight only, changes nothing
#   ./deploy.sh                    # copy + restart + verify
#   ./deploy.sh --rollback         # go back one release
#   ./deploy.sh --avatars anthony007
#
# Uses the same pinned interpreter as run.sh (grpcio + protobuf live in the app
# env, not the shell's python3). LT_PYTHON overrides it.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY="${LT_PYTHON:-/opt/miniconda3/envs/livetalking/bin/python}"
if [ ! -x "$PY" ]; then
  echo "deploy.sh: interpreter not found: $PY" >&2
  echo "deploy.sh: set LT_PYTHON, or create the env with ./setup.sh" >&2
  exit 1
fi
exec "$PY" scripts/deploy_remote.py "$@"
