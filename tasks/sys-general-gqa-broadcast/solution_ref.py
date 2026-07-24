import numpy as np


def gqa_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Grouped-query attention: query heads are split into n_kv contiguous
    groups of size g = n_q // n_kv, each group sharing one KV head.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n_q, n, d = Q.shape
    n_kv = K.shape[0]
    g = n_q // n_kv
    scale = 1.0 / np.sqrt(d)

    # Broadcast each KV head across its group of g query heads (blocked,
    # not interleaved: query head h uses KV head h // g).
    K_rep = np.repeat(K, g, axis=0)   # (n_q, n, d)
    V_rep = np.repeat(V, g, axis=0)   # (n_q, n, d)

    S = np.einsum("hnd,hmd->hnm", Q, K_rep) * scale
    S = S - np.max(S, axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / np.sum(P, axis=-1, keepdims=True)
    O = np.einsum("hnm,hmd->hnd", P, V_rep)
    return O
