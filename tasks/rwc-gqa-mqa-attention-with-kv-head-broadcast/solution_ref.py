import math
import numpy as np

def gqa_attention(Q, K, V):
    """Compute grouped-query attention.

    Args:
        Q: np.ndarray, shape (batch, seq_q, n_heads, head_dim)
        K: np.ndarray, shape (batch, seq_k, n_kv_heads, head_dim)
        V: np.ndarray, shape (batch, seq_k, n_kv_heads, head_dim)

    Returns:
        np.ndarray, shape (batch, seq_q, n_heads, head_dim), dtype float64
    """
    batch, seq_q, n_heads, head_dim = Q.shape
    n_kv_heads = K.shape[2]
    group_size = n_heads // n_kv_heads
    scale = head_dim ** -0.5

    out = np.zeros((batch, seq_q, n_heads, head_dim), dtype=np.float64)

    for b in range(batch):
        for i in range(seq_q):
            for h in range(n_heads):
                kv_h = h // group_size
                
                scores = [0.0] * K.shape[1]
                for k_idx in range(K.shape[1]):
                    dot = 0.0
                    for d in range(head_dim):
                        dot += Q[b, i, h, d] * K[b, k_idx, kv_h, d]
                    scores[k_idx] = dot * scale

                max_score = scores[0]
                for k_idx in range(1, len(scores)):
                    if scores[k_idx] > max_score:
                        max_score = scores[k_idx]

                exp_scores = [0.0] * len(scores)
                sum_exp = 0.0
                for k_idx in range(len(scores)):
                    val = math.exp(scores[k_idx] - max_score)
                    exp_scores[k_idx] = val
                    sum_exp += val

                weights = [0.0] * len(scores)
                for k_idx in range(len(scores)):
                    weights[k_idx] = exp_scores[k_idx] / sum_exp

                for d in range(head_dim):
                    weighted_sum = 0.0
                    for k_idx in range(K.shape[1]):
                        weighted_sum += weights[k_idx] * V[b, k_idx, kv_h, d]
                    out[b, i, h, d] = weighted_sum

    return out
