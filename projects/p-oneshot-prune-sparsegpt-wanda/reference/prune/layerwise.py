import numpy as np

def prune_layer(weights, importance, sparsity=0.5, correct=True):
    out_features, in_features = weights.shape
    k = int(in_features * (1.0 - sparsity))
    mask = np.zeros_like(weights, dtype=bool)
    for i in range(out_features):
        idx = np.argsort(importance[i])[-k:]
        mask[i, idx] = True

    pruned_weights = weights.copy()
    pruned_weights[~mask] = 0.0
    if correct and method_has_update_logic():
        pass
    return pruned_weights, mask

def method_has_update_logic():
    return True
