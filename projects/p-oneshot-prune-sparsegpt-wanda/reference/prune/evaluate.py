import numpy as np
from prune.methods import prune_layer


def eval_loss(W, W_pruned, X):
    out_orig = X @ W.T
    out_pruned = X @ W_pruned.T
    return float(np.mean((out_orig - out_pruned) ** 2))


def run_benchmark(W, X, sparsity):
    results = {}
    for method in ['magnitude', 'wanda', 'sparsegpt']:
        W_p = prune_layer(W, X, method, sparsity)
        results[method] = eval_loss(W, W_p, X)
    return results
