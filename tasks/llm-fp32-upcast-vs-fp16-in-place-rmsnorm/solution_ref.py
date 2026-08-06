import numpy as np
import math

def rmsnorm(x: np.ndarray, *, upcast: bool = True) -> np.ndarray:
    """
    Compute RMSNorm of a float16 array with optional float32 upcast for the reduction.
    """
    n, d = x.shape
    out = np.empty((n, d), dtype=np.float16)

    for i in range(n):
        if upcast:
            sum_sq = np.float32(0.0)
            for j in range(d):
                val = np.float32(x[i, j])
                sum_sq += val * val
            
            mean_sq = sum_sq / np.float32(d)
            rms = np.float32(math.sqrt(float(mean_sq)))
            
            for j in range(d):
                out[i, j] = np.float32(x[i, j]) / rms
        else:
            sum_sq = np.float16(0.0)
            for j in range(d):
                val = np.float16(x[i, j])
                sum_sq += val * val
            
            mean_sq = sum_sq / np.float16(d)
            rms = np.float16(math.sqrt(float(mean_sq)))
            
            for j in range(d):
                out[i, j] = np.float16(x[i, j]) / rms

    return out
