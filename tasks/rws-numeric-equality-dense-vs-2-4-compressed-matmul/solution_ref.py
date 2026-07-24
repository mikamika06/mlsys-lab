import numpy as np


def dense_vs_compressed24_matmul_error(W: np.ndarray, X: np.ndarray) -> float:
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    m, n = W.shape

    Y_dense = W @ X

    W_recon = np.zeros_like(W)
    n_groups = n // 4
    Wg = W.reshape(m, n_groups, 4)
    for g in range(n_groups):
        block = Wg[:, g, :]
        nz_rows, nz_cols = np.nonzero(block)
        W_recon[nz_rows, g * 4 + nz_cols] = block[nz_rows, nz_cols]

    Y_compressed = W_recon @ X
    return float(np.max(np.abs(Y_dense - Y_compressed)))
