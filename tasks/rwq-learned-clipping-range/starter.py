import numpy as np


def learned_clip_range(w: np.ndarray, group_size: int, bits: int,
                        n_steps: int = 25, lr: float = 0.05, eps: float = 1e-3):
    """Learn a per-group clip-range scalar alpha via sign-gradient descent.

    w: 1-D float64 array, len(w) a multiple of group_size.
    group_size: contiguous elements per group (each group gets its own alpha).
    bits: quantizer bit width, qmax = 2**(bits-1) - 1.
    n_steps, lr, eps: optimizer hyperparameters.

    For each contiguous group, starting from alpha = 1.0, repeat n_steps
    times: estimate d(MSE)/d(alpha) with a central finite difference of
    step `eps`, then alpha <- clip(alpha - lr*sign(grad), 0.2, 1.5), where
    MSE(alpha) is the mean squared error of symmetric qmax-level rounding
    of the group to clip value alpha * max(|group|).

    Returns (alphas, mses): float64 arrays of shape (len(w) // group_size,).
    """
    raise NotImplementedError('your code here')
