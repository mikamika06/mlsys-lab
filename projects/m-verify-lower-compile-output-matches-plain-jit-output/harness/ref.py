import re
import numpy as np

rng = np.random.default_rng(42)

AOT_OUTPUTS = [
    {"name": "out_0", "data": rng.normal(size=(8, 8)).astype(np.float32)},
    {"name": "out_1", "data": rng.normal(size=(4, 16)).astype(np.float32)},
]

JIT_OUTPUTS_MATCH = [
    {"name": "out_0", "data": AOT_OUTPUTS[0]["data"].copy()},
    {"name": "out_1", "data": AOT_OUTPUTS[1]["data"].copy()},
]

JIT_OUTPUTS_DRIFT = [
    {"name": "out_0", "data": AOT_OUTPUTS[0]["data"] + 1e-3},
    {"name": "out_1", "data": AOT_OUTPUTS[1]["data"].copy()},
]

STABLEHLO_IR_SAMPLE = """
module @jit_model attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<16x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<16x64xf32> {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %1 = stablehlo.exponential_minus_one %0 : tensor<16x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<16x64xf32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<16x64xf32>
    %4 = stablehlo.maximum %2, %3 : tensor<16x64xf32>
    return %4 : tensor<16x64xf32>
  }
}
"""

DESERIALIZED_OUTPUTS_MATCH = [
    {"name": "out_0", "data": AOT_OUTPUTS[0]["data"].copy()},
    {"name": "out_1", "data": AOT_OUTPUTS[1]["data"].copy()},
]


def verify_compile_vs_jit(aot_outputs, jit_outputs):
    max_err = 0.0
    for a, b in zip(aot_outputs, jit_outputs):
        err = float(np.max(np.abs(np.asarray(a["data"]) - np.asarray(b["data"]))))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err, "is_close": bool(max_err <= 1e-5)}


def analyze_stablehlo_ir(ir_text):
    op_pattern = re.compile(r"%[a-zA-Z0-9_%#:\.\-]+(?:\s*:\s*[^{=]+)?\s*=\s*(stablehlo\.[a-zA-Z0-9_]+)|^\s*(stablehlo\.[a-zA-Z0-9_]+)")
    counts = {}
    for line in ir_text.splitlines():
        line_str = line.strip()
        if not line_str or line_str.startswith("//"):
            continue
        match = op_pattern.search(line_str)
        if match:
            opname = match.group(1) or match.group(2)
            counts[opname] = counts.get(opname, 0) + 1
    return {"op_counts": counts, "unique_ops": sorted(counts.keys())}


def verify_serialized_numerics(original_outputs, deserialized_outputs, rtol=1e-5, atol=1e-5):
    all_matched = True
    max_err = 0.0
    for orig, des in zip(original_outputs, deserialized_outputs):
        a = np.asarray(orig["data"])
        b = np.asarray(des["data"])
        err = float(np.max(np.abs(a - b)))
        if err > max_err:
            max_err = err
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            all_matched = False
    return {"max_abs_err": max_err, "matches": all_matched}
