import math
import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Compute log-softmax along the last axis via the stable x − LSE identity."""
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    n_cols = orig_shape[-1]
    n_rows = 1
    for dim in orig_shape[:-1]:
        n_rows *= dim

    x_2d = x.reshape(n_rows, n_cols)
    out_2d = np.empty((n_rows, n_cols), dtype=np.float64)

    for i in range(n_rows):
        max_val = x_2d[i, 0]
        for j in range(1, n_cols):
            val = x_2d[i, j]
            if val > max_val:
                max_val = val

        sum_exp = 0.0
        for j in range(n_cols):
            sum_exp += math.exp(x_2d[i, j] - max_val)

        lse = max_val + math.log(sum_exp)

        for j in range(n_cols):
            out_2d[i, j] = x_2d[i, j] - lse

    return out_2d.reshape(orig_shape)
