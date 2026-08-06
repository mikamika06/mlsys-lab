import math
import numpy as np


def _chunk_partial(q, k_chunk, v_chunk):
    scale = 1.0 / math.sqrt(q.shape[0])
    
    # s = (k_chunk @ q) * scale
    # k_chunk is shape (chunk_len, d), q is shape (d,)
    chunk_len = k_chunk.shape[0]
    d = q.shape[0]
    s = np.empty((chunk_len,), dtype=np.float64)
    for i in range(chunk_len):
        dot_val = 0.0
        for j in range(d):
            dot_val += k_chunk[i, j] * q[j]
        s[i] = dot_val * scale

    # m_i = np.max(s)
    m_i = s[0]
    for i in range(1, chunk_len):
        if s[i] > m_i:
            m_i = s[i]

    # exp_s = np.exp(s - m_i)
    exp_s = np.empty((chunk_len,), dtype=np.float64)
    for i in range(chunk_len):
        exp_s[i] = math.exp(s[i] - m_i)

    # l_i = np.sum(exp_s)
    l_i = 0.0
    for i in range(chunk_len):
        l_i += exp_s[i]

    # o_i = (exp_s @ v_chunk) / l_i
    # exp_s is (chunk_len,), v_chunk is (chunk_len, v_dim)
    v_dim = v_chunk.shape[1]
    o_i = np.empty((v_dim,), dtype=np.float64)
    for j in range(v_dim):
        dot_v = 0.0
        for i in range(chunk_len):
            dot_v += exp_s[i] * v_chunk[i, j]
        o_i[j] = dot_v / l_i

    # L_i = m_i + np.log(l_i)
    L_i = m_i + math.log(l_i)

    return L_i, o_i


def two_chunk_split_kv_merge(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    N = k.shape[0]
    split = N // 2

    L1, O1 = _chunk_partial(q, k[:split], v[:split])
    L2, O2 = _chunk_partial(q, k[split:], v[split:])

    m = max(L1, L2)
    w1 = math.exp(L1 - m)
    w2 = math.exp(L2 - m)
    
    # return (O1 * w1 + O2 * w2) / (w1 + w2)
    v_dim = O1.shape[0]
    res = np.empty((v_dim,), dtype=np.float64)
    denom = w1 + w2
    for j in range(v_dim):
        res[j] = (O1[j] * w1 + O2[j] * w2) / denom
    return res
