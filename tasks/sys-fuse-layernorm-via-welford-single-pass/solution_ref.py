import math
import numpy as np


def layer_norm_welford(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    LayerNorm forward pass where the per-row mean and variance are
    computed with a single explicit pass over the D feature columns
    using Welford's online recurrence (vectorised across the batch
    dimension), instead of two separate reductions.

    x: (B, D) float64.
    gamma, beta: (D,) affine parameters.
    Returns: (B, D) normalized + affine output.
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    B, D = x.shape
    out = []

    for i in range(B):
        mean = 0.0
        M2 = 0.0
        for j in range(D):
            xj = x[i, j]
            count = j + 1
            delta = xj - mean
            mean = mean + delta / count
            delta2 = xj - mean
            M2 = M2 + delta * delta2

        var = M2 / D
        denom = math.sqrt(var + eps)

        row_out = []
        for j in range(D):
            xj = x[i, j]
            x_hat = (xj - mean) / denom
            val = gamma[j] * x_hat + beta[j]
            row_out.append(val)
        out.append(row_out)

    return np.array(out, dtype=np.float64)
