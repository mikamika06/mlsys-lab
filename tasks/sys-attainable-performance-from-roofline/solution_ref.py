import numpy as np
import math

def roofline_perf(ai, peak_flops, mem_bandwidth):
    """Return attainable performance per configuration."""
    ai_arr = np.asarray(ai, dtype=np.float64)
    peak_arr = np.asarray(peak_flops, dtype=np.float64)
    bw_arr = np.asarray(mem_bandwidth, dtype=np.float64)

    is_scalar = ai_arr.ndim == 0 and peak_arr.ndim == 0 and bw_arr.ndim == 0

    if is_scalar:
        val = min(float(peak_arr), float(ai_arr) * float(bw_arr))
        return np.array(val, dtype=np.float64)

    broadcasted = np.broadcast(ai_arr, peak_arr, bw_arr)
    result = np.empty(broadcasted.shape, dtype=np.float64)

    for idx, (a, p, b) in enumerate(broadcasted):
        result.flat[idx] = min(float(p), float(a) * float(b))

    return result
