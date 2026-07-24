import numpy as np


def sparsegpt_layerwise(W, X, sparsity, lam):
    W_work = np.asarray(W, dtype=np.float64).copy()
    m, d = W_work.shape

    H = 2.0 * X @ X.T + lam * np.eye(d)
    L = np.linalg.cholesky(H)
    Hinv = np.linalg.inv(L.T) @ np.linalg.inv(L)

    remove = int(m * d * sparsity)
    scores = (W_work * W_work) / (np.diag(Hinv)[None, :])
    remove_idx = np.argsort(scores.ravel())[:remove]

    mask = np.ones_like(W_work, dtype=bool)

    for idx in remove_idx:
        i, j = divmod(int(idx), d)
        if not mask[i, j]:
            continue
        value = W_work[i, j]
        mask[i, j] = False
        W_work[i, j] = 0.0
        W_work[i, :] += -value * Hinv[j, :] / Hinv[j, j]
        W_work[i, j] = 0.0

    return W_work, mask
