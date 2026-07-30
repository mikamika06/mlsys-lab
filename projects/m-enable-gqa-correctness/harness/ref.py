import numpy as np

CASES = [
    {"B": 2, "Hq": 8, "Hkv": 2, "L": 6, "S": 6, "D": 16, "is_causal": False, "seed": 0},
    {"B": 1, "Hq": 6, "Hkv": 3, "L": 5, "S": 5, "D": 8, "is_causal": True, "seed": 1},
    {"B": 3, "Hq": 12, "Hkv": 4, "L": 7, "S": 7, "D": 4, "is_causal": False, "seed": 2},
    {"B": 2, "Hq": 4, "Hkv": 4, "L": 5, "S": 5, "D": 8, "is_causal": True, "seed": 3},
    {"B": 1, "Hq": 9, "Hkv": 3, "L": 4, "S": 4, "D": 6, "is_causal": False, "seed": 4},
    {"B": 2, "Hq": 6, "Hkv": 1, "L": 5, "S": 5, "D": 8, "is_causal": True, "seed": 5},
]


def make_qkv(case):
    rng = np.random.default_rng(case["seed"])
    q = rng.standard_normal((case["B"], case["Hq"], case["L"], case["D"]))
    k = rng.standard_normal((case["B"], case["Hkv"], case["S"], case["D"]))
    v = rng.standard_normal((case["B"], case["Hkv"], case["S"], case["D"]))
    return q, k, v


def repeat_kv(x, n_rep):
    x = np.asarray(x)
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=-3)


def causal_bias(q_len, kv_len, dtype=np.float64):
    keep = np.tril(np.ones((q_len, kv_len), dtype=bool))
    bias = np.zeros((q_len, kv_len), dtype=dtype)
    bias[~keep] = -np.inf
    return bias


def scaled_dot_product_attention(query, key, value, is_causal=False, scale=None, enable_gqa=False):
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    head_dim = query.shape[-1]
    scale_factor = (1.0 / np.sqrt(head_dim)) if scale is None else scale
    if enable_gqa:
        n_rep = query.shape[-3] // key.shape[-3]
        key = repeat_kv(key, n_rep)
        value = repeat_kv(value, n_rep)
    q_len = query.shape[-2]
    kv_len = key.shape[-2]
    bias = causal_bias(q_len, kv_len) if is_causal else np.zeros((q_len, kv_len))
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale_factor
    scores = scores + bias
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, value)
