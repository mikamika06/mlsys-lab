import numpy as np

from .expand import repeat_kv
from .mask import causal_bias


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
    bias = causal_bias(q_len, kv_len, query.dtype) if is_causal else np.zeros((q_len, kv_len), dtype=query.dtype)
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale_factor
    scores = scores + bias
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, value)
