import numpy as np

def branchless_ops(x: np.ndarray, y: np.ndarray, mask: np.ndarray):
    """
    Element‑wise minimum/maximum/absolute/value selection without Python branches.
    Uses NumPy vectorised operations only.
    """
    mins = np.minimum(x, y)
    maxs = np.maximum(x, y)
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    sel = np.where(mask, x, y)
    return mins, maxs, abs_x, abs_y, sel
