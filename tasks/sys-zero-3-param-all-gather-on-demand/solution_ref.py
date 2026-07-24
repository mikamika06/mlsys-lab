import numpy as np


def zero3_linear_backward(weight_shards, x, grad_y):
    full_w = np.concatenate([np.asarray(s, dtype=np.float64) for s in weight_shards], axis=0)
    _ = x @ full_w.T
    full_grad = np.asarray(grad_y, dtype=np.float64).T @ np.asarray(x, dtype=np.float64)

    grads = []
    start = 0
    for shard in weight_shards:
        rows = shard.shape[0]
        grads.append(full_grad[start:start + rows].copy())
        start += rows
    return grads
