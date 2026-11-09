"""numpy <-> protobuf Tensor conversion (no pickle on the wire)."""
from __future__ import annotations

import numpy as np

from remote.proto import inference_pb2

_DTYPES = {
    "uint8": np.uint8,
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
}


def to_tensor(array: np.ndarray) -> inference_pb2.Tensor:
    """Pack a C-contiguous numpy array into a Tensor message."""
    arr = np.ascontiguousarray(array)
    kind = arr.dtype.name
    if kind not in _DTYPES:
        # widen to a wire-friendly dtype; latents etc. are float32/16 already
        arr = arr.astype(np.float32)
        kind = "float32"
    return inference_pb2.Tensor(
        dtype=kind,
        shape=list(arr.shape),
        data=arr.tobytes(),
    )


def from_tensor(message: inference_pb2.Tensor) -> np.ndarray:
    """Unpack a Tensor message into a numpy array (copy, writable)."""
    dtype = _DTYPES.get(message.dtype)
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype {message.dtype!r}")
    shape = tuple(int(d) for d in message.shape)
    arr = np.frombuffer(message.data, dtype=dtype)
    return arr.reshape(shape).copy()
