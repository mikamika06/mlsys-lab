import math
import numpy as np

def stable_softmax(logits):
    """Compute softmax along the last axis, numerically stable.

    Subtracts the per-row maximum before exponentiating to avoid overflow.
    """
    x = np.asarray(logits, dtype=np.float64)
    orig_shape = x.shape
    K = orig_shape[-1]
    flat_x = x.reshape(-1, K)
    N = flat_x.shape[0]
    out = np.empty((N, K), dtype=np.float64)
    for i in range(N):
        row = flat_x[i]
        m = row[0]
        for j in range(1, K):
            val = row[j]
            if val > m:
                m = val
        sum_e = 0.0
        e_row = [0.0] * K
        for j in range(K):
            val = math.exp(row[j] - m)
            e_row[j] = val
            sum_e += val
        for j in range(K):
            out[i, j] = e_row[j] / sum_e
    return out.reshape(orig_shape)
