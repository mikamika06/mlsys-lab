import numpy as np

from . import mapping


def expand_kv(kv, num_q_heads):
    num_kv_heads = kv.shape[0]
    idx = np.array(mapping.build_head_map(num_q_heads, num_kv_heads), dtype=np.int64)
    return kv[idx]


def gqa_attention(q, k, v, num_kv_heads):
    num_q_heads, seq_q, head_dim = q.shape
    k_exp = expand_kv(k, num_q_heads)
    v_exp = expand_kv(v, num_q_heads)
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.einsum("hqd,hkd->hqk", q, k_exp) * scale
    seq_k = k_exp.shape[1]
    mask = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1)
    scores = np.where(mask[None, :, :], -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,hkd->hqd", weights, v_exp)
