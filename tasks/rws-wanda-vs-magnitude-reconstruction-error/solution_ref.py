import numpy as np


def _masks(W, X, sparsity):
    d_out, d_in = W.shape
    n_prune = int(round(sparsity * d_in))
    col_norm = np.linalg.norm(X, axis=1)

    M_mag = np.ones_like(W)
    M_wanda = np.ones_like(W)
    for i in range(d_out):
        order_mag = np.argsort(np.abs(W[i]), kind="stable")
        M_mag[i, order_mag[:n_prune]] = 0.0

        metric = np.abs(W[i]) * col_norm
        order_w = np.argsort(metric, kind="stable")
        M_wanda[i, order_w[:n_prune]] = 0.0
    return M_wanda, M_mag


def _sq_frobenius_err(W, M, X):
    diff = W @ X - (W * M) @ X
    return float(np.sum(diff ** 2))


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
