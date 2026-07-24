import numpy as np


def obq_column_step(W, H_inv, col, scale, nmax):
    W = np.asarray(W, dtype=np.float64).copy()
    H_inv = np.asarray(H_inv, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)

    w_col = W[:, col]
    codes = np.clip(np.round(w_col / scale), -nmax, nmax)
    q_col = codes * scale

    err = (w_col - q_col) / H_inv[col, col]
    W[:, col] = q_col

    n = W.shape[1]
    if col + 1 < n:
        W[:, col + 1:] -= np.outer(err, H_inv[col, col + 1:])

    return q_col, W
