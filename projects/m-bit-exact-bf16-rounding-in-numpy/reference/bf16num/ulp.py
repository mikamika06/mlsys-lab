import numpy as np


def get_dtype_max(dtype_str: str) -> float:
    d = dtype_str.lower()
    if d in ("fp32", "float32"):
        return float(np.finfo(np.float32).max)
    if d in ("fp16", "float16"):
        return float(np.finfo(np.float16).max)
    if d in ("bf16", "bfloat16"):
        return float((2.0 - 2.0**-7) * (2.0**127))
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def compute_ulp(x: np.ndarray, dtype_str: str) -> np.ndarray:
    d = dtype_str.lower()
    if d in ("fp32", "float32"):
        m, min_exp = 23, -126
    elif d in ("fp16", "float16"):
        m, min_exp = 10, -14
    elif d in ("bf16", "bfloat16"):
        m, min_exp = 7, -126
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    arr = np.asarray(np.abs(x), dtype=np.float64)
    res = np.zeros_like(arr, dtype=np.float64)

    non_zero = arr > 0
    if np.any(non_zero):
        vals = arr[non_zero]
        exp = np.floor(np.log2(vals))
        exp = np.maximum(exp, min_exp)
        res[non_zero] = 2.0 ** (exp - m)

    res[~non_zero] = 2.0 ** (min_exp - m)
    return res
