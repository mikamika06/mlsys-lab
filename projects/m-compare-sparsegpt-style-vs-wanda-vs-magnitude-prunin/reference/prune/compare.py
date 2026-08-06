import numpy as np


def evaluate_pruning_methods(weights, activations, sparsity_ratio):
    W = weights.astype(np.float64)
    X = activations.astype(np.float64)

    n_features = W.shape[1]
    k = int(n_features * sparsity_ratio)

    mag_scores = np.abs(W)
    mag_mask = np.ones_like(W, dtype=bool)
    for i in range(W.shape[0]):
        idx = np.argsort(mag_scores[i])[:k]
        mag_mask[i, idx] = False
    W_mag = W * mag_mask

    X_norms = np.linalg.norm(X, axis=0, keepdims=True)
    wanda_scores = np.abs(W) * X_norms
    wanda_mask = np.ones_like(W, dtype=bool)
    for i in range(W.shape[0]):
        idx = np.argsort(wanda_scores[i])[:k]
        wanda_mask[i, idx] = False
    W_wanda = W * wanda_mask

    H = X.T @ X + 1e-4 * np.eye(n_features)
    H_inv = np.linalg.inv(H)
    sparsegpt_mask = np.ones_like(W, dtype=bool)
    W_sgpt = W.copy()
    for i in range(W.shape[0]):
        row = W[i]
        scores = np.abs(row) / np.sqrt(np.diag(H_inv))
        idx = np.argsort(scores)[:k]
        sparsegpt_mask[i, idx] = False

        err = row[idx] / np.diag(H_inv)[idx]
        W_sgpt[i] = row - np.outer(err, H_inv[idx, :]).sum(axis=0) / np.diag(H_inv)[idx, None]
        W_sgpt[i][~sparsegpt_mask[i]] = 0.0

    orig_out = W @ X.T
    results = {
        "magnitude": float(np.mean((W_mag @ X.T - orig_out) ** 2)),
        "wanda": float(np.mean((W_wanda @ X.T - orig_out) ** 2)),
        "sparsegpt": float(np.mean((W_sgpt @ X.T - orig_out) ** 2))
    }
    return results
