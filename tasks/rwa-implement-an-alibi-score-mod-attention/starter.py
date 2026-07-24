import numpy as np


def alibi_score_mod_attention(Q, K, V, slopes):
    """
    Q: (H, n, d) float array of queries per head.
    K: (H, m, d) float array of keys per head.
    V: (H, m, d_v) float array of values per head.
    slopes: (H,) float array of per-head ALiBi slopes.

    Returns the (H, n, d_v) float64 attention output after adding the
    ALiBi score_mod bias slopes[h] * (kv_idx - q_idx) to each scaled
    dot-product logit before the row-wise softmax.
    """
    raise NotImplementedError('your code here')
