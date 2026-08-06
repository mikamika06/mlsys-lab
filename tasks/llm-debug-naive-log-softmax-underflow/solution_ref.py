import math
import numpy as np

def log_softmax(x):
    """Compute log(softmax(x)) along the last axis (numerically stable)."""
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    last_dim = shape[-1]
    flat_x = x.reshape(-1, last_dim)
    num_rows = flat_x.shape[0]

    out_flat = np.empty((num_rows, last_dim), dtype=np.float64)

    for i in range(num_rows):
        row = flat_x[i]

        m = row[0]
        for j in range(1, last_dim):
            val = row[j]
            if val > m:
                m = val

        s = 0.0
        for j in range(last_dim):
            s += math.exp(row[j] - m)

        lse = math.log(s) + m

        for j in range(last_dim):
            out_flat[i, j] = row[j] - lse

    return out_flat.reshape(shape)
