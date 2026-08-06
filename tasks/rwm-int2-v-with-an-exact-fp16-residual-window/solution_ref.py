import math
import numpy as np


def kv_int2_residual_window(V: np.ndarray, group_size: int = 32, residual_window: int = 16) -> np.ndarray:
    """
    Quantize all but the last `residual_window` rows of `V` to 2 bits/element
    using grouped affine (zero-point) quantization along the channel axis;
    leave the last `residual_window` rows exact. Returns the reconstructed
    (T, d) array.
    """
    V = np.asarray(V, dtype=np.float64)
    T, d = V.shape
    Tq = T - residual_window

    Vq = V[:Tq]
    Vr = V[Tq:]

    ng = d // group_size

    lo = np.zeros((Tq, ng), dtype=np.float64)
    hi = np.zeros((Tq, ng), dtype=np.float64)
    scale = np.zeros((Tq, ng), dtype=np.float64)
    Vq_hat = np.zeros((Tq, d), dtype=np.float64)

    for i in range(Tq):
        for g in range(ng):
            start_col = g * group_size
            min_val = Vq[i, start_col]
            max_val = Vq[i, start_col]
            for j in range(1, group_size):
                val = Vq[i, start_col + j]
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
            lo[i, g] = min_val
            hi[i, g] = max_val
            s = (max_val - min_val) / 3.0
            if s == 0.0:
                s = 1.0
            scale[i, g] = s

    for i in range(Tq):
        for g in range(ng):
            start_col = g * group_size
            s = scale[i, g]
            l = lo[i, g]
            for j in range(group_size):
                col = start_col + j
                c = round((Vq[i, col] - l) / s)
                if c < 0.0:
                    c = 0.0
                elif c > 3.0:
                    c = 3.0
                Vq_hat[i, col] = c * s + l

    return np.concatenate([Vq_hat, Vr], axis=0)
