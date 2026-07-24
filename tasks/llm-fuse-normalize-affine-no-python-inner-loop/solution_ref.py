import numpy as np


def layernorm(x, gamma, beta, eps=1e-5):
    """Fused LayerNorm: normalize each row over the last axis, then apply the
    per-feature affine transform gamma/beta. Vectorized, no Python loops.
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta
