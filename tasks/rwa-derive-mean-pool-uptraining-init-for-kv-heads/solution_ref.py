import math
import numpy as np


def mean_pool_gqa_logit_mse(Q: np.ndarray, K: np.ndarray, n_rep: int) -> float:
    """
    Q: (n_heads, seq_q, d) original per-head queries.
    K: (n_heads, seq_k, d) original per-head keys (one MHA checkpoint).
    n_rep: number of original heads collapsed into each new shared GQA
    key head (n_heads must be divisible by n_rep).

    "Uptrain" the MHA checkpoint into GQA by MEAN-POOLING each contiguous
    group of n_rep K heads into one shared head, then let every head in
    that group attend against the shared, pooled key head instead of its
    own original key head. Return the mean squared error between those
    reconstructed attention logits and the original MHA logits.
    """
    n_heads, seq_q, d = Q.shape
    _, seq_k, _ = K.shape
    n_kv = n_heads // n_rep
    scale = 1.0 / math.sqrt(d)

    K_pooled = np.zeros((n_kv, seq_k, d), dtype=K.dtype)
    for kv_idx in range(n_kv):
        for sk_idx in range(seq_k):
            for d_idx in range(d):
                total = 0.0
                for r_idx in range(n_rep):
                    total += float(K[kv_idx * n_rep + r_idx, sk_idx, d_idx])
                K_pooled[kv_idx, sk_idx, d_idx] = total / float(n_rep)

    K_pooled_rep = np.zeros((n_heads, seq_k, d), dtype=K.dtype)
    for kv_idx in range(n_kv):
        for r_idx in range(n_rep):
            h_idx = kv_idx * n_rep + r_idx
            for sk_idx in range(seq_k):
                for d_idx in range(d):
                    K_pooled_rep[h_idx, sk_idx, d_idx] = K_pooled[kv_idx, sk_idx, d_idx]

    orig_logits = np.zeros((n_heads, seq_q, seq_k), dtype=Q.dtype)
    for h_idx in range(n_heads):
        for sq_idx in range(seq_q):
            for sk_idx in range(seq_k):
                dot_val = 0.0
                for d_idx in range(d):
                    dot_val += float(Q[h_idx, sq_idx, d_idx]) * float(K[h_idx, sk_idx, d_idx])
                orig_logits[h_idx, sq_idx, sk_idx] = dot_val * scale

    recon_logits = np.zeros((n_heads, seq_q, seq_k), dtype=Q.dtype)
    for h_idx in range(n_heads):
        for sq_idx in range(seq_q):
            for sk_idx in range(seq_k):
                dot_val = 0.0
                for d_idx in range(d):
                    dot_val += float(Q[h_idx, sq_idx, d_idx]) * float(K_pooled_rep[h_idx, sk_idx, d_idx])
                recon_logits[h_idx, sq_idx, sk_idx] = dot_val * scale

    sse_sum = 0.0
    count = 0
    for h_idx in range(n_heads):
        for sq_idx in range(seq_q):
            for sk_idx in range(seq_k):
                diff = float(recon_logits[h_idx, sq_idx, sk_idx]) - float(orig_logits[h_idx, sq_idx, sk_idx])
                sse_sum += diff * diff
                count += 1

    return float(sse_sum / count)
