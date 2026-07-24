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
    scale = 1.0 / np.sqrt(d)

    K_grouped = K.reshape(n_kv, n_rep, seq_k, d)
    K_pooled = K_grouped.mean(axis=1)              # (n_kv, seq_k, d)
    K_pooled_rep = np.repeat(K_pooled, n_rep, axis=0)  # (n_heads, seq_k, d)

    orig_logits = np.einsum("hqd,hkd->hqk", Q, K) * scale
    recon_logits = np.einsum("hqd,hkd->hqk", Q, K_pooled_rep) * scale

    return float(np.mean((recon_logits - orig_logits) ** 2))
