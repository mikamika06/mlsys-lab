import numpy as np


def wanda_score(W, X):
    X_norms = np.linalg.norm(X, axis=0)
    return np.abs(W) * X_norms


def sparsegpt_prune_row(w, H_inv, n_prune):
    w = w.copy()
    H_inv = H_inv.copy()
    mask = np.ones(len(w), dtype=bool)
    for _ in range(n_prune):
        scores = (w ** 2) / np.diag(H_inv)
        scores[~mask] = np.inf
        j = np.argmin(scores)
        mask[j] = False
        update = (w[j] / H_inv[j, j]) * H_inv[j, :]
        w = w - update
        w[j] = 0.0
        H_inv = H_inv - np.outer(H_inv[:, j], H_inv[j, :]) / H_inv[j, j]
    return w


def prune_layer(W, X, method, sparsity):
    W_pruned = W.copy()
    out_dim, in_dim = W.shape
    n_prune = int(in_dim * sparsity)
    if method == 'magnitude':
        scores = np.abs(W)
        for i in range(out_dim):
            idx = np.argsort(scores[i])[:n_prune]
            W_pruned[i, idx] = 0
    elif method == 'wanda':
        scores = wanda_score(W, X)
        for i in range(out_dim):
            idx = np.argsort(scores[i])[:n_prune]
            W_pruned[i, idx] = 0
    elif method == 'sparsegpt':
        H = X.T @ X
        damp = 0.01 * np.trace(H) / in_dim
        H = H + damp * np.eye(in_dim)
        H_inv = np.linalg.inv(H)
        for i in range(out_dim):
            W_pruned[i] = sparsegpt_prune_row(W[i], H_inv, n_prune)
    else:
        raise ValueError("Unknown method")
    return W_pruned
