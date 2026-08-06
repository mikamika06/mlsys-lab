import numpy as np
import math

def log1p_tiny(x: np.ndarray) -> np.ndarray:
    """
    Accurate computation of log(1+x) for tiny x.
    Uses NumPy's dedicated routine which handles catastrophic cancellation.
    """
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = math.log1p(x[i])
    return out
