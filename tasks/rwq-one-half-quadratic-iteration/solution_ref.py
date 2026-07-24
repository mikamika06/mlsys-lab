import numpy as np

EPS = 1e-8


def _shrink_lp(x, beta, p):
    ax = np.abs(x)
    if p == 1.0:
        mag = 1.0 / beta
    else:
        mag = (1.0 / beta) * np.power(ax + EPS, p - 1.0)
    return np.sign(x) * np.maximum(ax - mag, 0.0)


def hqq_half_quadratic_step(W: np.ndarray, s: np.ndarray, z: np.ndarray,
                             W_q: np.ndarray, lp: float, beta: float,
                             qmin: int, qmax: int):
    """
    One HQQ half-quadratic-splitting iteration, per-row zero-point groups.

    1. raw = W/s + z            (per-element target before rounding)
    2. r   = W_q - raw           (residual against the GIVEN codes W_q)
    3. W_e = shrink_lp(r, beta, lp)   (generalized-Lp shrinkage/soft-threshold)
    4. z_new = mean_over_row(W_q - W_e - W/s)   (least-squares zero-point)
    5. W_q_new = clip(round(W/s + z_new), qmin, qmax)   (re-quantize)

    s, z are shape (d_out,) -- one scalar per row (group = row).
    Returns (W_q_new, z_new).
    """
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    W_q = np.asarray(W_q, dtype=np.float64)

    raw = W / s[:, None] + z[:, None]
    r = W_q - raw
    W_e = _shrink_lp(r, beta, lp)
    z_new = np.mean(W_q - W_e - W / s[:, None], axis=1)
    W_q_new = np.clip(np.round(W / s[:, None] + z_new[:, None]), qmin, qmax)
    return W_q_new, z_new
