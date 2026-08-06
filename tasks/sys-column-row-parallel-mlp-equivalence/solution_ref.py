import math
import numpy as np


def _gelu_scalar(val):
    return 0.5 * val * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (val + 0.044715 * (val ** 3))))


def _gelu(x):
    x = np.asarray(x, dtype=np.float64)
    def apply_recursive(sub_arr):
        if sub_arr.ndim == 0:
            return _gelu_scalar(float(sub_arr))
        else:
            return [apply_recursive(sub_arr[i]) for i in range(sub_arr.shape[0])]
    res = apply_recursive(x)
    return np.asarray(res, dtype=np.float64)


def mlp_tensor_parallel(x, w1_shards, b1_shards, w2_shards, b2):
    """Column-parallel then row-parallel tensor-parallel MLP forward pass."""
    x = np.asarray(x, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    m = x.shape[0]
    d = x.shape[1]
    d_out = b2.shape[0]
    y_data = [[0.0 for _ in range(d_out)] for _ in range(m)]
    for w1_shard, b1_shard, w2_shard in zip(w1_shards, b1_shards, w2_shards):
        w1 = np.asarray(w1_shard, dtype=np.float64)
        b1 = np.asarray(b1_shard, dtype=np.float64)
        w2 = np.asarray(w2_shard, dtype=np.float64)
        h = w1.shape[1]
        prod1 = [[0.0 for _ in range(h)] for _ in range(m)]
        for i in range(m):
            for j in range(h):
                s = 0.0
                for k in range(d):
                    s += x[i, k] * w1[k, j]
                prod1[i][j] = s + b1[j]
        a = _gelu(prod1)
        prod2 = [[0.0 for _ in range(d_out)] for _ in range(m)]
        for i in range(m):
            for j in range(d_out):
                s = 0.0
                for k in range(h):
                    s += a[i, k] * w2[k, j]
                prod2[i][j] = s
        for i in range(m):
            for j in range(d_out):
                y_data[i][j] += prod2[i][j]
    for i in range(m):
        for j in range(d_out):
            y_data[i][j] += b2[j]
    return np.asarray(y_data, dtype=np.float64)
