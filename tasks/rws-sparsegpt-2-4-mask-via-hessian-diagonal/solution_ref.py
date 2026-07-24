import numpy as np


def sparsegpt_24_prune(W: np.ndarray, X: np.ndarray, damp: float = 0.01):
    """SparseGPT-style 2:4 structured pruning: Hessian-diagonal mask
    selection + one-shot OBS/OBC compensation update.

    See task.md for the exact formulas. Returns (mask, W_hat).
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    O, I = W.shape

    H = X.T @ X
    H = H + damp * np.mean(np.diag(H)) * np.eye(I)
    Hinv = np.linalg.inv(H)
    diag_hinv = np.diag(Hinv)

    mask = np.ones_like(W, dtype=np.int64)
    for o in range(O):
        for g0 in range(0, I, 4):
            idx = np.arange(g0, g0 + 4)
            scores = W[o, idx] ** 2 / diag_hinv[idx]
            order = np.argsort(scores, kind="stable")
            prune = idx[order[:2]]
            mask[o, prune] = 0

    W_hat = W.copy()
    for o in range(O):
        S = np.where(mask[o] == 0)[0]
        w_S = W[o, S]
        Hinv_SS = Hinv[np.ix_(S, S)]
        delta = -(Hinv[:, S] @ np.linalg.solve(Hinv_SS, w_S))
        W_hat[o] = W[o] + delta
        W_hat[o, S] = 0.0

    return mask, W_hat
