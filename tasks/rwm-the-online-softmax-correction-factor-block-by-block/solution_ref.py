import numpy as np


def online_softmax_blocks(scores, values, block_size):
    scores = np.asarray(scores, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    m_old = -np.inf
    l_old = 0.0
    a_old = 0.0
    rows = []

    for start in range(0, len(scores), block_size):
        s = scores[start:start + block_size]
        v = values[start:start + block_size]

        m_block = np.max(s)
        e = np.exp(s - m_block)
        l_block = np.sum(e)
        a_block = np.sum(e * v)

        m_new = max(m_old, m_block)
        old_scale = 0.0 if np.isneginf(m_old) else np.exp(m_old - m_new)
        block_scale = np.exp(m_block - m_new)

        l_old = l_old * old_scale + l_block * block_scale
        a_old = a_old * old_scale + a_block * block_scale
        m_old = m_new

        rows.append([m_old, l_old, a_old])

    return np.asarray(rows, dtype=np.float64)
