import numpy as np
from numpy.lib.stride_tricks import as_strided


def sliding_window(x: np.ndarray, w: int) -> np.ndarray:
    """Zero-copy (N-w+1, w) view of all length-w contiguous windows of 1D `x`."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    n = x.shape[0]
    w = int(w)
    if w < 1 or w > n:
        raise ValueError(f"window {w} out of range for length {n}")
    s = x.strides[0]
    return as_strided(x, shape=(n - w + 1, w), strides=(s, s))
