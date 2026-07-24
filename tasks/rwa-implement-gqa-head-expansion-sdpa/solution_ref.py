import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def gqa_head_expansion_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n_q = Q.shape[2]
    n_kv = K.shape[2]
    n_rep = n_q // n_kv

    # repeat_interleave: each KV head repeated n_rep times consecutively.
    K_exp = np.repeat(K, n_rep, axis=2)
    V_exp = np.repeat(V, n_rep, axis=2)

    d = Q.shape[-1]
    Qh = Q.transpose(0, 2, 1, 3)
    Kh = K_exp.transpose(0, 2, 1, 3)
    Vh = V_exp.transpose(0, 2, 1, 3)

    scores = (Qh @ Kh.swapaxes(-2, -1)) / np.sqrt(d)
    weights = _softmax(scores, axis=-1)
    out = (weights @ Vh).transpose(0, 2, 1, 3)

    memory_ratio = n_kv / n_q
    return out, memory_ratio
