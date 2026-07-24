import numpy as np


def gqa_broadcast_attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    H_q, n, d = q.shape
    H_kv = k.shape[0]
    n_rep = H_q // H_kv

    k_rep = np.repeat(k, n_rep, axis=0)
    v_rep = np.repeat(v, n_rep, axis=0)

    scale = 1.0 / np.sqrt(d)
    scores = np.matmul(q, k_rep.transpose(0, 2, 1)) * scale
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=-1, keepdims=True)
    return np.matmul(w, v_rep)
