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
        self.shape_of = {}
        self.outputs = []
        self.inputs = []
        out_specs = []
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            shp = tuple(self.engine.get_tensor_shape(nm))
            dt = _torch_dtype(self.engine.get_tensor_dtype(nm))
            if self.engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
                self.in_dt[nm] = dt
                self.shape_of[nm] = shp
                self.inputs.append((nm, shp))
            else:
                out_specs.append((nm, shp, dt))
        for nm, shp, dt in out_specs:
            buf = torch.empty(shp, dtype=dt, device=device).contiguous()
            self.ctx.set_tensor_address(nm, buf.data_ptr())
            self.outputs.append((nm, buf))
        self._keep = []

    def run(self, feeds, keep_out=None):
        keep = keep_out if keep_out is not None else self._keep
        if keep_out is None:
            keep[:] = list(feeds.values())
        for nm, t in feeds.items():
            # TRT does NOT validate tensor sizes against the engine's static
            # shapes — a wrong-sized buffer is read as far as the engine
            # needs and silently yields garbage. Check here: a mismatched
            # lane (e.g. 64-edge features into a 32-edge engine) must fail
            # loudly, not render nonsense at a great frame rate.
            want = self.shape_of[nm]
            if tuple(t.shape) != want:
                raise ValueError(
                    f"engine input {nm} wants {want}, got {tuple(t.shape)}")
            t = t.to(self.in_dt[nm]).contiguous()
            keep.append(t)
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
        self._resolve_io_names()
        logger.info("[ditto] hybrid warp (TRT 10, plugin-free) from %s "
                    "(feature edge %d)", engine_dir, self.feature_edge)

    def _resolve_io_names(self):
        """Derive the cut-point tensor names from the engines themselves.

        The 512 lane was built from upstream's shipped ONNX (GridSample
        node names); engines exported from torch (the 256 lane) carry
        GridSample3D names. Keying by role rather than by literal string
        serves both — and any future re-export.
        """
        base = {"feature_3d", "kp_source", "kp_driving"}
        b_inputs = set(self.b.in_dt)
        cut = b_inputs - base
        assert len(cut) == 1, f"segB has unexpected inputs: {sorted(b_inputs)}"
        self.gs1_in = cut.pop()

        a_outs = [nm for nm, _ in self.a.outputs]
        assert len(a_outs) == 2, f"segA outputs: {a_outs}"
        # vol is the (bs*(1+kp), 4, D, H, W) source pyramid; grid ends in 3.
        vol_shp = self._shape_of(self.a.outputs, a_outs[0])
        self.a_out_vol, self.a_out_grid = (
            (a_outs[0], a_outs[1]) if vol_shp[-1] != 3
            else (a_outs[1], a_outs[0]))

        outs_b = {nm: shp for nm, shp in ((nm, buf.shape) for nm, buf in self.b.outputs)}
        occ = [nm for nm, shp in outs_b.items() if len(shp) == 4]
        deform = [nm for nm, shp in outs_b.items() if len(shp) == 5]
        assert len(occ) == 1 and len(deform) == 1, f"segB outputs: {outs_b}"
        self.b_out_occ = occ[0]
        self.b_out_deform = deform[0]

        c_inputs = [nm for nm in self.c.in_dt]
        assert self.b_out_occ in c_inputs and len(c_inputs) == 2, \
            f"segC inputs {c_inputs} do not carry segB's occlusion"
        self.gs2_out = next(nm for nm in c_inputs if nm != self.b_out_occ)

        gs2_shp = dict(self.c.inputs)[self.gs2_out]
        self.feature_edge = int(gs2_shp[-1])

    @staticmethod
    def _shape_of(pairs, name):
        for nm, buf in pairs:
            if nm == name:
                return tuple(buf.shape)
        raise KeyError(name)

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
            gs1 = self._grid_sample(a_out[self.a_out_vol], a_out[self.a_out_grid])
            b_feeds = dict(base)
            b_feeds[self.gs1_in] = gs1
            b_out = self.b.run(b_feeds)
            occ = b_out[self.b_out_occ].clone()
            warped = self._grid_sample(f3d, b_out[self.b_out_deform])
            c_out = self.c.run({self.gs2_out: warped, self.b_out_occ: occ})
            torch.cuda.current_stream().synchronize()
            return c_out["out"].clone().cpu().numpy()

    # ── device-resident batch lane ────────────────────────────────────────
    # The live GPU half runs the 10-frame window as 10 independent warp
    # chains. The sync/D2H above is per-chain — 10 pipeline drains per batch,
    # and decode then copies the result back to the device. The batch lane
    # removes exactly that: all 10 chains enqueue back-to-back on the SAME
    # stream (kernel order alone makes the shared engine output buffers
    # safe), ONE sync before decode, zero host round-trips.
    #
    # Numerics: identical kernels, identical order, identical inputs — the
    # only difference is WHEN the host waits. Buffer-reuse hazard: each
    # chain's inputs are kept alive and its outputs CONSUMED (grid-sampled
    # into fresh tensors) before the next chain overwrites the engine's
    # static output buffers, so a chain never reads a later chain's data.

    def run_batch(self, batches, device_out=True):
        """batches: iterable of (feature_3d, kp_source, kp_driving) numpy.

        Enqueues every chain on the current stream and returns the list of
        (B, 256, E, E) device tensors (device_out=True) or numpy (False).
        device_out=True does NOT sync: the caller consumes on the same
        stream (decode), whose own sync is the batch's single wait point.
        Returned tensors are fresh copies (clone/float), never the engines'
        static output buffers."""
        outs = []
        keep = []
        with TRTWarpHybrid._lock:
            for feature_3d, kp_source, kp_driving in batches:
                f3d = torch.from_numpy(
                    np.ascontiguousarray(feature_3d)).to(self.device)
                kps = torch.from_numpy(
                    np.ascontiguousarray(kp_source)).to(self.device)
                kpd = torch.from_numpy(
                    np.ascontiguousarray(kp_driving)).to(self.device)
                base = {"feature_3d": f3d, "kp_source": kps,
                        "kp_driving": kpd}
                a_out = self.a.run(base, keep)
                gs1 = self._grid_sample(
                    a_out[self.a_out_vol], a_out[self.a_out_grid])
                b_feeds = dict(base)
                b_feeds[self.gs1_in] = gs1
                b_out = self.b.run(b_feeds, keep)
                occ = b_out[self.b_out_occ]
                warped = self._grid_sample(f3d, b_out[self.b_out_deform])
                c_out = self.c.run({self.gs2_out: warped,
                                    self.b_out_occ: occ}, keep)
                out = c_out["out"]
                # the engine's output buffer is reused by the next chain:
                # materialise a fresh tensor NOW (stream-ordered) so the
                # returned entries never alias each other.
                outs.append(out.float() if out.dtype != torch.float32
                            else out.clone())
            del keep[:]  # feeds only need to live until their kernels run
        if device_out:
            return outs
        torch.cuda.current_stream().synchronize()
        return [t.cpu().numpy() for t in outs]
