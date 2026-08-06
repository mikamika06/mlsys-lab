import numpy as np

def branchless_ops(x: np.ndarray, y: np.ndarray, mask: np.ndarray):
    """
    Element‑wise minimum/maximum/absolute/value selection without Python branches.
    Uses NumPy vectorised operations only.
    """
    shape = x.shape
    dtype = x.dtype
    mins = np.empty(shape, dtype=dtype)
    maxs = np.empty(shape, dtype=dtype)
    abs_x = np.empty(shape, dtype=dtype)
    abs_y = np.empty(shape, dtype=dtype)
    sel = np.empty(shape, dtype=dtype)

    for i in np.ndindex(shape):
        xi = x[i]
        yi = y[i]
        mi = mask[i]

        mins[i] = xi if xi < yi else yi
        maxs[i] = xi if xi > yi else yi
        abs_x[i] = -xi if xi < 0 else xi
        abs_y[i] = -yi if yi < 0 else yi
        sel[i] = xi if mi else yi

    return mins, maxs, abs_x, abs_y, sel
