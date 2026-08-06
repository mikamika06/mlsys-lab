"""Overflow and underflow calculation module."""

import numpy as np


def compute_overflow_underflow_fractions(tensor: np.ndarray, dtype_str: str) -> dict:
    """Compute exact overflow and underflow fractions for a tensor given a target format ('fp16' or 'bf16')."""
    arr = np.asarray(tensor, dtype=np.float64)
    total = arr.size
    if total == 0:
        return {"overflow": 0.0, "underflow": 0.0}

    if dtype_str.lower() in ("fp16", "float16"):
        max_finite = 65504.0
        min_pos_normal = 6.103515625e-5
    elif dtype_str.lower() in ("bf16", "bfloat16"):
        max_finite = 3.3895313892515355e38
        min_pos_normal = 1.1754943508222875e-38
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    abs_arr = np.abs(arr)
    overflow_count = np.sum(abs_arr > max_finite)
    underflow_count = np.sum((abs_arr > 0) & (abs_arr < min_pos_normal))

    return {
        "overflow": float(overflow_count / total),
        "underflow": float(underflow_count / total),
    }
