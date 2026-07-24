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
    raise NotImplementedError('your code here')
