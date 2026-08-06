import numpy as np


def wanda_prune(weights, activations, sparsity_ratio, domain_shift=False):
    W = weights.astype(np.float64)
    X = activations.astype(np.float64)

    if domain_shift:
        X = X * 0.1 + np.random.default.normal(0, 5.0, size=X.shape)

    X_norms = np.linalg.norm(X, axis=0, keepdims=True)
    scores = np.abs(W) * X_norms

    n_features = W.shape[1]
    k = int(n_features * sparsity_ratio)
    mask = np.ones_like(W, dtype=bool)
    for i in range(W.shape[0]):
        idx = np.argsort(scores[i])[:k]
        mask[i, idx] = False

    W_pruned = W * mask
    err = float(np.mean((W_pruned @ X.T - W @ X.T) ** 2))
    return {"pruned_weights": W_pruned, "error": err, "norms": X_norms}
