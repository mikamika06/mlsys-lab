import math
import numpy as np

def log1p_tiny(x: np.ndarray) -> np.ndarray:
    """
    Accurate computation of log(1+x) for tiny x.
    Uses NumPy's built‑in log1p which is stable for |x| << 1.
    """
    out = np.empty(x.shape, dtype=np.float64)
    for i in range(x.size):
        out.flat[i] = math.log1p(float(x.flat[i]))
    return out
