import math
import numpy as np


def stable_softplus(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty(x_arr.shape, dtype=np.float64)
    out_flat = out.flat
    for i, val in enumerate(x_arr.flat):
        v = float(val)
        if math.isnan(v):
            out_flat[i] = math.nan
        else:
            max_val = v if v > 0.0 else 0.0
            out_flat[i] = max_val + math.log1p(math.exp(-abs(v)))
    return out
