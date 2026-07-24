import numpy as np


def layernorm(x, gamma, beta, eps=1e-5):
    """Fused LayerNorm over the last axis, then per-feature affine (gamma, beta).

    Args:
        x: float array of shape (N, D).
        gamma: scale, shape (D,).
        beta: shift, shape (D,).
        eps: variance epsilon.

    Returns:
        float64 array of shape (N, D). Vectorized NumPy only, no Python loops.
    """
    raise NotImplementedError("your code here")
