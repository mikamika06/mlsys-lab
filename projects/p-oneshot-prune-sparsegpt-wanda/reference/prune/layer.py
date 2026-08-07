import numpy as np

def prune_unstructured(w, scores, sparsity):
    out_features, in_features = w.shape
    k = max(1, int(in_features * (1.0 - sparsity)))
    mask = np.zeros_like(w, dtype=bool)

    for i in range(out_features):
        idx = np.argsort(scores[i])[-k:]
        mask[i, idx] = True

    w_pruned = w.copy()
    w_pruned[~mask] = 0.0
    return w_pruned, mask

def correct_bias(w, w_pruned, x):
    x_mean = np.mean(x, axis=1)
    return (w - w_pruned) @ x_mean
