import math
import numpy as np


def _mse_at_alpha(group: np.ndarray, alpha: float, qmax: int) -> float:
    amax = 0.0
    for i in range(group.shape[0]):
        val = group[i]
        abs_val = val if val >= 0.0 else -val
        if abs_val > amax:
            amax = abs_val
            
    c = alpha * amax
    if c <= 1e-12:
        c = 1e-12
    scale = c / qmax

    total_sq_err = 0.0
    n = group.shape[0]
    for i in range(n):
        val = group[i]
        div = val / scale
        if div >= 0.0:
            rounded = math.floor(div + 0.5)
        else:
            rounded = math.ceil(div - 0.5)
            
        if rounded > qmax:
            codes = float(qmax)
        elif rounded < -qmax:
            codes = float(-qmax)
        else:
            codes = float(rounded)
            
        ghat = codes * scale
        diff = ghat - val
        total_sq_err += diff * diff

    return total_sq_err / float(n)


def learned_clip_range(w: np.ndarray, group_size: int, bits: int,
                        n_steps: int = 25, lr: float = 0.05, eps: float = 1e-3):
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
            
            if grad > 0.0:
                sign_grad = 1.0
            elif grad < 0.0:
                sign_grad = -1.0
            else:
                sign_grad = 0.0
                
            alpha = alpha - lr * sign_grad
            if alpha < 0.2:
                alpha = 0.2
            elif alpha > 1.5:
                alpha = 1.5
                
        alphas[g] = alpha
        mses[g] = _mse_at_alpha(seg, alpha, qmax)

    return alphas, mses
