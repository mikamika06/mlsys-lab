import numpy as np


def _oracle(W, X, sparsity, block_size):
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
        if count > 0:
            flat = scores.ravel()
            order = np.argsort(flat)
            chosen = order[:count]
            rows, cols = np.unravel_index(chosen, scores.shape)
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


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [[0.8, -1.1, 0.5, 2.0],
                 [1.5, 0.3, -0.7, 0.2],
                 [-0.4, 1.2, 0.9, -1.3]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 0.2, -0.3, 0.7],
                 [-0.5, 1.1, 0.4, -0.8],
                 [0.6, -0.9, 1.2, 0.3],
                 [0.1, 0.5, -0.6, 1.0]],
                dtype=np.float64,
            ),
            0.5,
            2,
        ),
        (
            np.array(
                [[1.0, -2.0, 3.0, 4.0, -1.0],
                 [0.5, 1.5, -2.5, 0.8, 2.2]],
                dtype=np.float64,
            ),
            np.array(
                [[0.4, 0.7],
                 [1.0, -0.2],
                 [-0.3, 0.9],
                 [0.6, -0.8],
                 [1.1, 0.5]],
                dtype=np.float64,
            ),
            0.4,
            3,
        ),
    ]

    best = 0.0
    for W, X, sparsity, block_size in cases:
        try:
            got = sol.sparsegpt_prune(W, X, sparsity, block_size)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _oracle(W, X, sparsity, block_size)
        err = np.linalg.norm(got @ X - ref @ X) / (
            np.linalg.norm(ref @ X) + 1e-12
        )
        best = max(best, float(err))
    return {"rel_err": best}
