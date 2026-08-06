import numpy as np

def obs_prune(W, X, sparsity):
    N, d = X.shape
    H = (X.T @ X) / N
    damp = 0.01 * np.trace(H) / d
    H += np.eye(d) * damp
    H_inv = np.linalg.inv(H)

    k = int(d * sparsity)
    W_new = np.zeros_like(W)
    mask = np.ones_like(W, dtype=bool)

    if k == 0:
        return W.copy(), mask

    for i in range(W.shape[0]):
        w = W[i].copy()
        for _ in range(k):
            diag = np.diag(H_inv)
            score = (w**2) / diag
            score[~mask[i]] = np.inf
            j = np.argmin(score)
            mask[i, j] = False
            w -= w[j] * H_inv[j, :] / H_inv[j, j]
            w[j] = 0.0
        W_new[i] = w

    return W_new, mask
