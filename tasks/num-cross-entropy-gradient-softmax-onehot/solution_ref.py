import math
import numpy as np


def cross_entropy_backward(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Gradient of the mean softmax cross-entropy w.r.t. ``logits``."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = z.shape[0]
    c = z.shape[1]

    out = np.zeros((n, c), dtype=np.float64)

    for i in range(n):
        row_max = float(z[i, 0])
        for j in range(1, c):
            val = float(z[i, j])
            if val > row_max:
                row_max = val

        e_sum = 0.0
        for j in range(c):
            val = math.exp(float(z[i, j]) - row_max)
            out[i, j] = val
            e_sum += val

        target = int(y[i])
        for j in range(c):
            p = out[i, j] / e_sum
            if j == target:
                p -= 1.0
            out[i, j] = p / float(n)

    return out
