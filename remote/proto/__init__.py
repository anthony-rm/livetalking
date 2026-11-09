"""Generated protobuf/gRPC stubs for the remote worker.

Regenerate after editing inference.proto:

    python -m grpc_tools.protoc -I remote/proto \
        --python_out=remote/proto --grpc_python_out=remote/proto \
        remote/proto/inference.proto

Then re-apply the relative import in inference_pb2_grpc.py:
    from . import inference_pb2 as inference__pb2
"""
