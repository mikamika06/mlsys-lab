import numpy as np

FORMAT_TESTS = [
    ("FP32", {"sign_bits": 1, "exp_bits": 8, "mantissa_bits": 23}),
    ("TF32", {"sign_bits": 1, "exp_bits": 8, "mantissa_bits": 10}),
    ("FP16", {"sign_bits": 1, "exp_bits": 5, "mantissa_bits": 10}),
]

SWEEP_LAYERS = [
    np.array([0.123, 0.456, 0.789], dtype=np.float32),
    np.array([1.5, 2.5, 3.5], dtype=np.float32),
]

PREDICTOR_CASES = [
    ({"max_val": 10.0, "min_val": -10.0, "op_type": "MatMul"}, {"force_fp32": True}, "FP32"),
    ({"max_val": 1e6, "min_val": -1e6, "op_type": "MatMul"}, {"force_fp32": False}, "FP32"),
    ({"max_val": 1.0, "min_val": -1.0, "op_type": "MatMul", "allow_fp16": True}, {"force_fp32": False, "prefer_tf32": True}, "TF32"),
]
