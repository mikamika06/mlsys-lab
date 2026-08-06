import numpy as np


def online_softmax_update(
    m_old,
    l_old,
    O_old,
    m_block,
    l_block,
    O_block,
):
    m_new = max(m_old, m_block)
    alpha = np.exp(m_old - m_new)
    beta = np.exp(m_block - m_new)

    l_new = alpha * l_old + beta * l_block
    O_new = (
        alpha * l_old * np.asarray(O_old, dtype=np.float64)
        + beta * l_block * np.asarray(O_block, dtype=np.float64)
    ) / l_new

    return float(m_new), float(l_new), np.asarray(O_new, dtype=np.float64)
