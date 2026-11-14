"""Plugin-free, deterministic TensorRT-10 hybrid warp for Ditto.

Ditto's `warp_network` graph contains two 5-D GridSample ops. TensorRT has no
native 5-D GridSample, and the custom `grid_sample_3d` plugin is built against
the TRT 8 ABI (won't even load on TRT 10) and is non-deterministic under
concurrent GPU load. Instead of the plugin we split the graph at the two
GridSample nodes into three plugin-free TRT 10 engines and run the two 5-D
bilinear grid samples in PyTorch fp32 between them (contention-immune):

    feature_3d,kp_s,kp_d --[segA TRT]--> gs1 vol,grid
    --[torch grid_sample 5D]--> gs1_out
    --[segB TRT]--> deform grid, occlusion
    --[torch grid_sample 5D on feature_3d]--> warped
    --[segC TRT]--> out (1,256,64,64)

Engines come from scripts/ditto/warp_split_segments.py + onnx2trt10-style build
(see the repo's PLAN / scratch harness). Technique from
https://github.com/RemiEtien/liveportrait-trt10-hybrid-warp (MIT).

Runtime design (matters — the naive TRTWrapper path was SLOWER than pytorch):
pre-allocated output buffers, inputs fed as torch CUDA tensors,
`execute_async_v3` on the CURRENT torch stream (ordered with the grid samples,
no cross-stream race), ONE synchronize per chain, and a process-wide lock (a TRT
execution context is not thread-safe and DittoReal may overlap idle-tick renders
with talk renders).

Caller contract matches upstream WarpNetwork: numpy in, numpy out.
"""
import logging
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

GS1_IN = "/dense_motion_network/GridSample_output_0"
A_OUT_VOL = "/dense_motion_network/Reshape_3_output_0"
A_OUT_GRID = "/dense_motion_network/Reshape_4_output_0"
B_OUT_DEFORM = "/dense_motion_network/Transpose_1_output_0"
B_OUT_OCC = "/dense_motion_network/Sigmoid_output_0"
GS2_OUT = "/GridSample_output_0"

DEFAULT_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "ditto-weights", "warp_hybrid_engines")


def _torch_dtype(trt_dt):
    import tensorrt as trt
    return {trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16}.get(trt_dt, torch.float32)


class _Engine:
    """Plugin-free TRT engine with pre-allocated outputs; runs on the current
    torch stream; no per-call allocation, no per-call synchronize."""

    def __init__(self, path, logger, device="cuda"):
        import tensorrt as trt
        if not os.path.exists(path):
            raise FileNotFoundError(f"hybrid warp engine not found: {path}")
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_dt = {}
        self.outputs = []
        out_specs = []
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            shp = tuple(self.engine.get_tensor_shape(nm))
            dt = _torch_dtype(self.engine.get_tensor_dtype(nm))
            if self.engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
                self.in_dt[nm] = dt
            else:
                out_specs.append((nm, shp, dt))
        for nm, shp, dt in out_specs:
            buf = torch.empty(shp, dtype=dt, device=device).contiguous()
            self.ctx.set_tensor_address(nm, buf.data_ptr())
            self.outputs.append((nm, buf))
        self._keep = []

    def run(self, feeds):
        self._keep = list(feeds.values())
        for nm, t in feeds.items():
            t = t.to(self.in_dt[nm]).contiguous()
            self._keep.append(t)
            self.ctx.set_tensor_address(nm, t.data_ptr())
        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream_handle=stream.cuda_stream)
        return dict(self.outputs)


class TRTWarpHybrid:
    _lock = threading.Lock()

    def __init__(self, engine_dir=None, device="cuda"):
        import tensorrt as trt
        self.device = device
        engine_dir = engine_dir or os.environ.get("DITTO_WARP_ENGINE_DIR",
                                                  DEFAULT_ENGINE_DIR)
        trt_logger = trt.Logger(trt.Logger.ERROR)
        self.a = _Engine(os.path.join(engine_dir, "hyb_segA.engine"), trt_logger, device)
        self.b = _Engine(os.path.join(engine_dir, "hyb_segB.engine"), trt_logger, device)
        self.c = _Engine(os.path.join(engine_dir, "hyb_segC.engine"), trt_logger, device)
        logger.info("[ditto] hybrid warp (TRT 10, plugin-free) from %s", engine_dir)

    @staticmethod
    def _grid_sample(vol, grid):
        return F.grid_sample(vol.float(), grid.float(), mode="bilinear",
                             padding_mode="zeros", align_corners=False)

    def __call__(self, feature_3d, kp_source, kp_driving):
        f3d = torch.from_numpy(np.ascontiguousarray(feature_3d)).to(self.device)
        kps = torch.from_numpy(np.ascontiguousarray(kp_source)).to(self.device)
        kpd = torch.from_numpy(np.ascontiguousarray(kp_driving)).to(self.device)

        with TRTWarpHybrid._lock:
            base = {"feature_3d": f3d, "kp_source": kps, "kp_driving": kpd}
            a_out = self.a.run(base)
            gs1 = self._grid_sample(a_out[A_OUT_VOL], a_out[A_OUT_GRID])
            b_feeds = dict(base)
            b_feeds[GS1_IN] = gs1
            b_out = self.b.run(b_feeds)
            occ = b_out[B_OUT_OCC].clone()
            warped = self._grid_sample(f3d, b_out[B_OUT_DEFORM])
            c_out = self.c.run({GS2_OUT: warped, B_OUT_OCC: occ})
            torch.cuda.current_stream().synchronize()
            return c_out["out"].clone().cpu().numpy()
