import numpy as np


def enable_gqa_broadcast_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n_q = Q.shape[1]
    n_kv = K.shape[1]
    r = n_q // n_kv

    # Blocked broadcast: query heads [k*r, (k+1)*r) all read KV head k.
    K_exp = np.repeat(K, r, axis=1)
    V_exp = np.repeat(V, r, axis=1)

    d = Q.shape[-1]
    scores = (Q @ K_exp.swapaxes(-2, -1)) / np.sqrt(d)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    return weights @ V_exp
