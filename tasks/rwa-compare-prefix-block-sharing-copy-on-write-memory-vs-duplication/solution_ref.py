import numpy as np


def _causal_attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    d = q.shape[-1]
    n = q.shape[0]
    scores = (q @ k.T) / np.sqrt(d)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=-1, keepdims=True)
    return w @ v


def cow_prefix_attention(q_a, k_a, v_a, q_b, k_b, v_b, shared_prefix_len, block_size):
    len_a = np.asarray(q_a).shape[0]
    len_b = np.asarray(q_b).shape[0]

    blocks_a = -(-len_a // block_size)
    blocks_b = -(-len_b // block_size)
    shared_blocks = min(shared_prefix_len // block_size, blocks_a, blocks_b)

    duplicated = blocks_a + blocks_b
    unique = duplicated - shared_blocks
    size_ratio = float(duplicated / unique) if unique else 0.0

    out_a = _causal_attention(q_a, k_a, v_a)
    out_b = _causal_attention(q_b, k_b, v_b)

    return size_ratio, out_a, out_b
