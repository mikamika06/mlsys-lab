import numpy as np


def merge_online_softmax(m_a, l_a, o_a, m_b, l_b, o_b):
    m_new = np.maximum(m_a, m_b)
    alpha = np.exp(m_a - m_new)
    beta = np.exp(m_b - m_new)
    l_new = alpha * l_a + beta * l_b
    o_new = (alpha[:, None] * l_a[:, None] * o_a + beta[:, None] * l_b[:, None] * o_b) / l_new[:, None]
    return m_new, l_new, o_new


def chunked_online_attention(q, k, v, chunk_size=64):
    seq_len_k, d_k = k.shape
    d_v = v.shape[1]
    batch_size = q.shape[0]
    scale = 1.0 / np.sqrt(d_k)

    m_cum = np.full((batch_size,), -np.inf, dtype=np.float64)
    l_cum = np.zeros((batch_size,), dtype=np.float64)
    o_cum = np.zeros((batch_size, d_v), dtype=np.float64)

    for i in range(0, seq_len_k, chunk_size):
        k_chunk = k[i:i + chunk_size]
        v_chunk = v[i:i + chunk_size]

        scores = (q @ k_chunk.T) * scale
        m_b = np.max(scores, axis=-1)
        scores_shift = scores - m_b[:, None]
        exp_scores = np.exp(scores_shift)
        l_b = np.sum(exp_scores, axis=-1)
        o_b = (exp_scores @ v_chunk) / l_b[:, None]

        if i == 0:
            m_cum, l_cum, o_cum = m_b, l_b, o_b
        else:
            m_cum, l_cum, o_cum = merge_online_softmax(m_cum, l_cum, o_cum, m_b, l_b, o_b)

    return o_cum
