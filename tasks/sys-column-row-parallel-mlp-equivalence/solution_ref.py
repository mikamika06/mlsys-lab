import numpy as np


def _gelu(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def mlp_tensor_parallel(x, w1_shards, b1_shards, w2_shards, b2):
    """Column-parallel then row-parallel tensor-parallel MLP forward pass."""
    x = np.asarray(x, dtype=np.float64)
    partials = []
    for w1, b1, w2 in zip(w1_shards, b1_shards, w2_shards):
        w1 = np.asarray(w1, dtype=np.float64)
        b1 = np.asarray(b1, dtype=np.float64)
        w2 = np.asarray(w2, dtype=np.float64)
        a = _gelu(x @ w1 + b1)
        partials.append(a @ w2)
    y = np.sum(partials, axis=0) + np.asarray(b2, dtype=np.float64)
    return y
