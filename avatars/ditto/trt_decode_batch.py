"""Static-B decoder engine for Ditto's SPADE decoder (TensorRT 10, fp16).

B is the LMDM window (10 online frames per pass) — the engine is FROZEN at
B=10 on purpose: static-shape kernels are TRT's fastest, and the LMDM emits
exactly 10 frames per pass so the whole window takes ONE engine call
instead of ten (measured 31.8 -> 25.9 ms/frame, and one synchronize per
window instead of ten).

A partial window (< B) is padded with the last real frame and the pads are
dropped from the result — outputs for real frames are unaffected (verified:
batch-independence exact, mean|diff| vs the per-frame engine 0.78/255).

Runtime rules from trt_warp_hybrid (load-bearing there, same here):
pre-allocated IO bound ONCE via set_tensor_address, inputs copied into the
resident buffers, execute_async_v3 on the CURRENT torch stream, one
synchronize per call.
"""
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TRTDecodeBatch:
    """(B,256,64,64) -> (B,3,512,512) fp32 on device, sigmoid range 0..1."""

    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"batched decoder engine not found: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.device = device
        self.batch = tuple(self.engine.get_tensor_shape("feature"))[0]
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            shp = tuple(self.engine.get_tensor_shape(nm))
            dt = {trt.DataType.FLOAT: torch.float32,
                  trt.DataType.HALF: torch.float16}[self.engine.get_tensor_dtype(nm)]
            buf = torch.empty(shp, dtype=dt, device=device)
            self.ctx.set_tensor_address(nm, buf.data_ptr())
            if nm == "feature":
                self.in_buf = buf
            else:
                self.out_buf = buf
        logger.info("[ditto] batched decoder B=%d: %s", self.batch, engine_path)

    def __call__(self, feature):
        t = feature if torch.is_tensor(feature) else torch.from_numpy(
            np.ascontiguousarray(feature))
        n = t.shape[0]
        assert n <= self.batch, f"decode batch {n} > engine {self.batch}"
        if n < self.batch:
            t = torch.cat([t, t[-1:].repeat(self.batch - n, 1, 1, 1)])
        self.in_buf.copy_(t.to(self.in_buf.dtype), non_blocking=True)
        self.ctx.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        return self.out_buf.float()[:n]
