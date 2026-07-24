import numpy as np


def reconstruct_pruned_weights(W: np.ndarray, X: np.ndarray, prune_order: list[int]) -> np.ndarray:
    W_hat = np.asarray(W, dtype=np.float64).copy()
    X = np.asarray(X, dtype=np.float64)
    n = W_hat.shape[1]

    H = X @ X.T + 1e-6 * np.eye(n, dtype=np.float64)

    L = np.linalg.cholesky(H)
    I = np.eye(n, dtype=np.float64)
    Hinv = np.linalg.solve(L.T, np.linalg.solve(L, I))

    states = []
    for pos, q in enumerate(prune_order):
        removed = W_hat[:, q].copy()
        W_hat[:, q] = 0.0
        for j in prune_order[pos + 1:]:
            W_hat[:, j] += removed * (Hinv[q, j] / Hinv[q, q])
        states.append(W_hat.copy())

    return np.stack(states, axis=0)
