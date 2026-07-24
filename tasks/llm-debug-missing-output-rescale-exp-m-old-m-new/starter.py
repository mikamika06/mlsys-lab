import numpy as np


def online_softmax_update(
    m_old,
    l_old,
    O_old,
    m_block,
    l_block,
    O_block,
):
    # TODO: missing the exp(m_old - m_new) rescale on the old accumulator.
    m_new = max(m_old, m_block)
    beta = np.exp(m_block - m_new)

    l_new = np.exp(m_old - m_new) * l_old + beta * l_block
    O_new = (
        l_old * np.asarray(O_old, dtype=np.float64)
        + beta * l_block * np.asarray(O_block, dtype=np.float64)
    ) / l_new

    return float(m_new), float(l_new), O_new
