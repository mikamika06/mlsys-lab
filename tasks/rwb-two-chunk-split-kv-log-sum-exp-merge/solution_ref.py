import math


def _chunk_partial(q, k_chunk, v_chunk):
    d = len(q)
    scale = 1.0 / math.sqrt(d)

    chunk_len = len(k_chunk)
    v_dim = len(v_chunk[0]) if chunk_len > 0 else 0

    s = [0.0] * chunk_len
    for i in range(chunk_len):
        dot_val = 0.0
        for j in range(d):
            dot_val += k_chunk[i][j] * q[j]
        s[i] = dot_val * scale

    m_i = s[0]
    for i in range(1, chunk_len):
        if s[i] > m_i:
            m_i = s[i]

    exp_s = [0.0] * chunk_len
    for i in range(chunk_len):
        exp_s[i] = math.exp(s[i] - m_i)

    l_i = 0.0
    for i in range(chunk_len):
        l_i += exp_s[i]

    o_i = [0.0] * v_dim
    for j in range(v_dim):
        dot_v = 0.0
        for i in range(chunk_len):
            dot_v += exp_s[i] * v_chunk[i][j]
        o_i[j] = dot_v / l_i

    L_i = m_i + math.log(l_i)

    return L_i, o_i


def two_chunk_split_kv_merge(q, k, v):
    N = len(k)
    split = N // 2

    L1, O1 = _chunk_partial(q, k[:split], v[:split])
    L2, O2 = _chunk_partial(q, k[split:], v[split:])

    m = max(L1, L2)
    w1 = math.exp(L1 - m)
    w2 = math.exp(L2 - m)

    v_dim = len(O1)
    res = [0.0] * v_dim
    denom = w1 + w2
    for j in range(v_dim):
        res[j] = (O1[j] * w1 + O2[j] * w2) / denom
    return res
