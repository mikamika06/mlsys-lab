import numpy as np


def scaled_dot_product_attention(q, k, v):
    # TODO: fix the attention scale. This version omits the required
    # 1/sqrt(head_dim) normalization, causing overly sharp softmax weights.
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    logits = q @ k.T
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ v
