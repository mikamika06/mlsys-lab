import math
import numpy as np

def generalized_soft_threshold(x: np.ndarray, beta: float, p: float) -> np.ndarray:
    """Generalized soft‑thresholding for the $L_p$ quasi‑norm ($0<p\le1$)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        val = x[i]
        abs_val = val if val >= 0.0 else -val
        if abs_val == 0.0:
            thresh = 0.0
        else:
            thresh = beta * math.pow(abs_val, p - 1.0)
        diff = abs_val - thresh
        shrunk_abs = diff if diff > 0.0 else 0.0
        sign_val = 1.0 if val > 0.0 else (-1.0 if val < 0.0 else 0.0)
        out[i] = sign_val * shrunk_abs
    return out
