import numpy as np

def log1p_tiny(x: np.ndarray) -> np.ndarray:
    """
    Accurate computation of log(1+x) for tiny x.
    Uses NumPy's built‑in log1p which is stable for |x| << 1.
    """
    return np.log1p(x)
