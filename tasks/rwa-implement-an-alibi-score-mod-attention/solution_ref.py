import numpy as np


def alibi_score_mod_attention(Q, K, V, slopes):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)

    H, n, d = Q.shape
    m = K.shape[1]

    scores = np.einsum("hnd,hmd->hnm", Q, K) / np.sqrt(d)

    q_idx = np.arange(n, dtype=np.float64)[:, None]
    kv_idx = np.arange(m, dtype=np.float64)[None, :]
    bias = kv_idx - q_idx  # (n, m): score_mod's (kv_idx - q_idx) term

    scores = scores + slopes[:, None, None] * bias[None, :, :]

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    return np.einsum("hnm,hmv->hnv", weights, V)
