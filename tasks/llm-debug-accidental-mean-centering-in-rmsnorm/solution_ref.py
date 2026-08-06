import math
import numpy as np


def rms_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Normalized array with the same shape and dtype float64."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)

    shape = x.shape
    d = shape[-1]
    n_rows = 1
    for dim in shape[:-1]:
        n_rows *= dim

    x_2d = x.reshape((n_rows, d))
    out_2d = out.reshape((n_rows, d))

    for i in range(n_rows):
        s = 0.0
        for j in range(d):
            v = x_2d[i, j]
            s += v * v
        rms = math.sqrt((s / d) + eps)
        for j in range(d):
            out_2d[i, j] = x_2d[i, j] / rms

    return out
