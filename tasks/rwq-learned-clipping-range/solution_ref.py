import numpy as np


def _mse_at_alpha(group: np.ndarray, alpha: float, qmax: int) -> float:
    amax = float(np.max(np.abs(group)))
    c = alpha * amax
    if c <= 1e-12:
        c = 1e-12
    scale = c / qmax
    codes = np.clip(np.round(group / scale), -qmax, qmax)
    ghat = codes * scale
    return float(np.mean((ghat - group) ** 2))


def learned_clip_range(w: np.ndarray, group_size: int, bits: int,
                        n_steps: int = 25, lr: float = 0.05, eps: float = 1e-3):
    """Learn a per-group clip-range scalar alpha via sign-gradient descent.

    See task.md for the full derivation. For each contiguous group of
    `group_size` elements of `w`:
      1. alpha = 1.0
      2. n_steps times: central-finite-difference the group's rounding
         MSE(alpha) at step eps, then alpha <- clip(alpha - lr*sign(grad),
         0.2, 1.5).
      3. Record the final alpha and MSE(alpha) at that final alpha.

    Returns (alphas, mses), each float64 arrays of shape
    (len(w) // group_size,).
    """
    w = np.asarray(w, dtype=np.float64)
    n_groups = w.shape[0] // group_size
    qmax = (1 << (bits - 1)) - 1

    alphas = np.zeros(n_groups, dtype=np.float64)
    mses = np.zeros(n_groups, dtype=np.float64)

    for g in range(n_groups):
        seg = w[g * group_size:(g + 1) * group_size]
        alpha = 1.0
        for _ in range(n_steps):
            f_plus = _mse_at_alpha(seg, alpha + eps, qmax)
            f_minus = _mse_at_alpha(seg, alpha - eps, qmax)
            grad = (f_plus - f_minus) / (2.0 * eps)
            alpha = alpha - lr * float(np.sign(grad))
            alpha = min(max(alpha, 0.2), 1.5)
        alphas[g] = alpha
        mses[g] = _mse_at_alpha(seg, alpha, qmax)

    return alphas, mses
