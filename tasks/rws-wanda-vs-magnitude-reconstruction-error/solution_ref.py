import math
import numpy as np


def _masks(W, X, sparsity):
    d_out, d_in = W.shape
    n = X.shape[1]
    n_prune = int(round(sparsity * d_in))
    col_norm = np.zeros(d_in, dtype=X.dtype)
    for j in range(d_in):
        s = 0.0
        for k in range(n):
            s += X[j, k] ** 2
        col_norm[j] = math.sqrt(s)

    M_mag = np.ones_like(W)
    M_wanda = np.ones_like(W)
    for i in range(d_out):
        indexed_mag = [(math.fabs(W[i, j]), j) for j in range(d_in)]
        order_mag = [j for _, j in sorted(indexed_mag, key=lambda x: x[0])]
        for idx in order_mag[:n_prune]:
            M_mag[i, idx] = 0.0

        indexed_w = [(math.fabs(W[i, j]) * col_norm[j], j) for j in range(d_in)]
        order_w = [j for _, j in sorted(indexed_w, key=lambda x: x[0])]
        for idx in order_w[:n_prune]:
            M_wanda[i, idx] = 0.0
    return M_wanda, M_mag


def _sq_frobenius_err(W, M, X):
    d_out, d_in = W.shape
    n = X.shape[1]
    total = 0.0
    for i in range(d_out):
        for k in range(n):
            wx = 0.0
            wmx = 0.0
            for j in range(d_in):
                wx += W[i, j] * X[j, k]
                wmx += W[i, j] * M[i, j] * X[j, k]
            diff = wx - wmx
            total += diff * diff
    return float(total)


def wanda_vs_magnitude_error(W: np.ndarray, X: np.ndarray, sparsity: float):
    """
    For the same per-row sparsity level, compare two pruning-mask
    criteria by the squared Frobenius output-reconstruction error they
    cause on the SAME calibration activations X:

    - Wanda:      importance_ij = |W_ij| * ||X[j, :]||_2   (activation-aware)
    - Magnitude:  importance_ij = |W_ij|                    (weight-only)

    Each row prunes the lowest-`sparsity`-fraction entries by its own
    criterion (stable sort, ties broken by column index). Error for a
    mask M: ||W@X - (W*M)@X||_F^2.

    Returns (e_wanda, e_magnitude).
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    M_wanda, M_mag = _masks(W, X, sparsity)
    e_wanda = _sq_frobenius_err(W, M_wanda, X)
    e_mag = _sq_frobenius_err(W, M_mag, X)
    return e_wanda, e_mag
