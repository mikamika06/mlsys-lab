import numpy as np

def log1p_tiny(x: np.ndarray) -> np.ndarray:
    """
    Accurate computation of log(1+x) for tiny x.
    Uses NumPy's dedicated routine which handles catastrophic cancellation.
    """
    return np.log1p(x)
