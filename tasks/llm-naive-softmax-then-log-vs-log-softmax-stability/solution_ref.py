import math
import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable log‑softmax along the last axis."""
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    last_dim = shape[-1]
    out_shape = shape[:-1] + (1,)
    out = np.zeros(out_shape, dtype=np.float64)

    x_flat = np.reshape(x, (-1, last_dim))
    out_flat = np.reshape(out, (-1, 1))

    num_rows = x_flat.shape[0]
    for i in range(num_rows):
        mx = x_flat[i, 0]
        for j in range(1, last_dim):
            val = x_flat[i, j]
            if val > mx:
                mx = val

        s = 0.0
        for j in range(last_dim):
            s += math.exp(x_flat[i, j] - mx)

        out_flat[i, 0] = -mx + math.log(s)

    return out
