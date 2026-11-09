#!/usr/bin/env bash
# Regenerate the remote-worker gRPC stubs from remote/proto/inference.proto.
# Requires grpcio-tools in the conda env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
PY="${LT_PYTHON:-/opt/miniconda3/envs/livetalking/bin/python}"
"$PY" -m grpc_tools.protoc -I remote/proto \
    --python_out=remote/proto --grpc_python_out=remote/proto \
    remote/proto/inference.proto
# The generator emits an absolute import; make it package-relative so
# `import remote.proto.inference_pb2_grpc` works.
"$PY" - "$SCRIPT_DIR/remote/proto/inference_pb2_grpc.py" <<'EOF'
import sys, io
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
s = s.replace("import inference_pb2 as inference__pb2",
              "from . import inference_pb2 as inference__pb2")
io.open(p, "w", encoding="utf-8").write(s)
EOF
echo "regenerated remote/proto/inference_pb2{,_grpc}.py"
