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
    mean = np.zeros(B, dtype=np.float64)
    M2 = np.zeros(B, dtype=np.float64)

    for j in range(D):
        xj = x[:, j]
        count = j + 1
        delta = xj - mean
        mean = mean + delta / count
        delta2 = xj - mean
        M2 = M2 + delta * delta2

    var = M2 / D
    x_hat = (x - mean[:, None]) / np.sqrt(var[:, None] + eps)
    return gamma * x_hat + beta
