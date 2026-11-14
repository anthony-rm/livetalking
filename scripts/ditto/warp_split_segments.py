"""Split Ditto's warp_network.onnx into 3 plugin-free sub-graphs at the two
GridSample3D boundaries (identical structure to liveportrait-trt10-hybrid-warp).

segA: [feature_3d, kp_source, kp_driving] -> gs1 vol/grid
segB: [feature_3d, kp_source, kp_driving, gs1_out] -> [gs2_grid, occlusion]
segC: [gs2_out, occlusion] -> out
"""
import argparse
import os
import onnx
from onnx import shape_inference
from onnx.utils import Extractor

DM = "/dense_motion_network"

SEGMENTS = {
    "hyb_segA": {
        "inputs": ["feature_3d", "kp_source", "kp_driving"],
        "outputs": [f"{DM}/Reshape_3_output_0", f"{DM}/Reshape_4_output_0"],
    },
    "hyb_segB": {
        "inputs": ["feature_3d", "kp_source", "kp_driving", f"{DM}/GridSample_output_0"],
        "outputs": [f"{DM}/Transpose_1_output_0", f"{DM}/Sigmoid_output_0"],
    },
    "hyb_segC": {
        "inputs": ["/GridSample_output_0", f"{DM}/Sigmoid_output_0"],
        "outputs": ["out"],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.onnx}...")
    model = onnx.load(args.onnx)
    model = shape_inference.infer_shapes(model)
    ext = Extractor(model)

    for name, io in SEGMENTS.items():
        out_path = os.path.join(args.out_dir, f"{name}.onnx")
        print(f"Extracting {name}: {io['inputs']} -> {io['outputs']}")
        sub = ext.extract_model(io["inputs"], io["outputs"])
        sub = shape_inference.infer_shapes(sub)
        onnx.checker.check_model(sub)
        onnx.save(sub, out_path)
        print(f"  [ok] saved {out_path} ({len(sub.graph.node)} nodes)")


if __name__ == "__main__":
    main()
