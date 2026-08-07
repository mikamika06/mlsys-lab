import numpy as np


def compute_reference_attention(q, k, v, causal=False):
    seq_len, dim = q.shape
    scale = 1.0 / np.sqrt(dim)
    scores = np.matmul(q, k.T) * scale
    if causal:
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    probs = exp_scores / sum_exp
    return np.matmul(probs, v)
