import numpy as np


def _shrink_op(x: np.ndarray, beta: float, p: float) -> np.ndarray:
    """Proximal operator of beta * ||.||_p^p (elementwise soft/half-quadratic shrink)."""
    if p == 1.0:
        return np.sign(x) * np.maximum(np.abs(x) - 1.0 / beta, 0.0)
    return np.sign(x) * np.maximum(np.abs(x) - (1.0 / beta) * np.abs(x) ** (p - 1.0), 0.0)


def hqq_optimize(W, scale, zero0, qmin, qmax, lp_norm, beta0, kappa, iters):
    """
    HQQ-style zero-point optimization: scale is held fixed; the zero-point z
    is refined for `iters` half-quadratic passes, then W is quantized one
    final time with the converged z. Returns (W_q, z, W_dequant).
    """
    W = np.asarray(W, dtype=np.float64)
    zero = float(zero0)
    beta = float(beta0)

    for _ in range(iters):
        W_q = np.clip(np.round(W * scale + zero), qmin, qmax)
        W_r = (W_q - zero) / scale
        W_e = _shrink_op(W - W_r, beta, lp_norm)
        zero = float(np.mean(W_q - (W - W_e) * scale))
        beta *= kappa

    W_q = np.clip(np.round(W * scale + zero), qmin, qmax)
    W_dq = (W_q - zero) / scale
    return W_q, zero, W_dq
