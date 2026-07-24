import numpy as np


def sparsegpt_prune(W, X, sparsity, block_size):
    W = np.asarray(W, dtype=np.float64).copy()
    m, n = W.shape
    k = X.shape[1]

    H = (X @ X.T) / k + 1e-4 * np.eye(n, dtype=np.float64)
    Hinv = np.linalg.inv(H)

    mask = np.zeros_like(W, dtype=bool)

    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        scores = W[:, start:end] ** 2 / Hinv.diagonal()[start:end][None, :]
        count = int(scores.size * sparsity)
        if count:
            selected = np.argsort(scores.ravel())[:count]
            rows, cols = np.unravel_index(selected, scores.shape)
            mask[rows, start + cols] = True

    for q in range(n):
        for row in range(m):
            if mask[row, q]:
                value = W[row, q]
                W[row, q] = 0.0
                W[row, q + 1:] -= (
                    value / Hinv[q, q]
                ) * Hinv[q, q + 1:]

    return W
