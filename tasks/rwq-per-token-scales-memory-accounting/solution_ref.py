import numpy as np
import math

def compute_scales_and_size(K: np.ndarray, V: np.ndarray):
    """
    Compute per‑row absolute‑max scales for K and V and the memory size ratio.
    """
    n, dK = K.shape
    _, dV = V.shape

    scales_K = np.empty(n, dtype=np.float64)
    for i in range(n):
        max_val = 0.0
        for j in range(dK):
            val = K[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        scales_K[i] = max_val

    scales_V = np.empty(n, dtype=np.float64)
    for i in range(n):
        max_val = 0.0
        for j in range(dV):
            val = V[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        scales_V[i] = max_val

    orig_bytes = (n * dK + n * dV) * 4
    quant_bytes = (n * dK + n * dV) * 1 + n * 4 * 2
    size_ratio = orig_bytes / quant_bytes

    return scales_K, scales_V, float(size_ratio)
