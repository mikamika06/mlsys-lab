import numpy as np


def causal_bias(q_len, kv_len, dtype=np.float64):
    keep = np.tril(np.ones((q_len, kv_len), dtype=bool))
    bias = np.zeros((q_len, kv_len), dtype=dtype)
    bias[~keep] = -np.inf
    return bias
