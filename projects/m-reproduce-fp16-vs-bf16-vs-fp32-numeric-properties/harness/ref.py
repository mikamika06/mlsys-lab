import math
import numpy as np

TEST_DTYPES = ["fp16", "bf16", "fp32"]

FORMAT_REF = {
    "fp16": {
        "exponent_bits": 5,
        "mantissa_bits": 10,
        "max_val": 65504.0,
        "min_pos_normal": 2.0**(-14),
        "min_pos_subnormal": 2.0**(-24),
        "eps": 2.0**(-10),
    },
    "bf16": {
        "exponent_bits": 8,
        "mantissa_bits": 7,
        "max_val": (2.0 - 2.0**(-7)) * (2.0**127),
        "min_pos_normal": 2.0**(-126),
        "min_pos_subnormal": 2.0**(-133),
        "eps": 2.0**(-7),
    },
    "fp32": {
        "exponent_bits": 8,
        "mantissa_bits": 23,
        "max_val": (2.0 - 2.0**(-23)) * (2.0**127),
        "min_pos_normal": 2.0**(-126),
        "min_pos_subnormal": 2.0**(-149),
        "eps": 2.0**(-23),
    },
}

AUTOCAST_BENCHMARK_OPS = {
    "matmul": "cast",
    "linear": "cast",
    "conv2d": "cast",
    "softmax": "keep_fp32",
    "layer_norm": "keep_fp32",
    "cross_entropy": "keep_fp32",
    "add": "promote",
    "mul": "promote",
}

STDS_TO_CHECK = [100.0, 1000.0, 5000.0, 10000.0, 20000.0, 30000.0]


def get_ref_properties(dtype_str: str) -> dict:
    return dict(FORMAT_REF[dtype_str])


def ref_fp16_overflow_prob(std: float) -> float:
    if std <= 0.0:
        return 0.0
    z = 65504.0 / std
    return math.erfc(z / math.sqrt(2.0))
