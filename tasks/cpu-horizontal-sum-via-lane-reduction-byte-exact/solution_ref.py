import numpy as np

def lane_reduce_sum(a: np.ndarray) -> np.ndarray:
    """
    Compute the horizontal sum of a 1‑D integer array via SIMD‑style lane reduction.
    The result is returned as a scalar NumPy array with the same dtype as *a*.
    """
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Input must be a one‑dimensional integer array.")
    total = np.sum(a, dtype=a.dtype)
    return np.array(total, dtype=a.dtype)
