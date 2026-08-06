import math
import numpy as np

def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    out = np.empty_like(x, dtype=x.dtype)
    for i in range(x.shape[0]):
        val = x[i]
        abs_val = val if val >= 0.0 else -val
        max_val = 0.0 if 0.0 >= val else val
        exp_val = math.exp(-abs_val)
        log1p_val = math.log1p(exp_val)
        out[i] = max_val + log1p_val
    return out
