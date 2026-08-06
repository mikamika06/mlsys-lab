import numpy as np
import math

def quantize_fixed_point(arr: np.ndarray, frac_bits: int) -> np.ndarray:
    """Quantizes a float array to fixed‑point with round‑half‑to‑even."""
    scale = 1 << frac_bits
    out = np.empty(arr.shape, dtype=np.int64)
    for i in range(arr.size):
        val = float(arr.flat[i]) * scale
        f = math.floor(val)
        frac = val - f
        if frac > 0.5:
            res = f + 1
        elif frac < 0.5:
            res = f
        else:
            if f % 2 == 0:
                res = f
            else:
                res = f + 1
        out.flat[i] = int(res)
    return out
