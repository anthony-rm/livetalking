#!/usr/bin/env bash
# LiveTalking launcher — uses the conda env created by setup.sh
# Usage: ./run.sh [extra app.py args]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Explicit env — don't let the active conda prefix override it.
# Override with LT_PYTHON if the env lives somewhere other than /opt/miniconda3.
PY="${LT_PYTHON:-/opt/miniconda3/envs/livetalking/bin/python}"
if [ ! -x "$PY" ]; then
  echo "run.sh: interpreter not found: $PY" >&2
  echo "run.sh: set LT_PYTHON, or create the env with ./setup.sh" >&2
  exit 1
fi
exec "$PY" app.py "$@"