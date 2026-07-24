import numpy as np


def sparsegpt_2_4(W: np.ndarray, X: np.ndarray):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    H = (X.T @ X) / X.shape[0] + 1e-4 * np.eye(X.shape[1], dtype=np.float64)
    Hinv = np.linalg.inv(H)

    mask = np.zeros(W.shape, dtype=np.int64)
    W_hat = W.copy()

    for r in range(W.shape[0]):
        for start in range(0, W.shape[1], 4):
            cols = range(start, start + 4)
            ranked = sorted(
                cols,
                key=lambda c: (W[r, c] ** 2) / Hinv[c, c]
            )
            for c in ranked[2:]:
                mask[r, c] = 1

            for c in ranked[:2]:
                old = W_hat[r, c]
                for k in range(start, start + 4):
                    if mask[r, k]:
                        W_hat[r, k] -= old * Hinv[k, c] / Hinv[c, c]
                W_hat[r, c] = 0.0

    return mask, W_hat
