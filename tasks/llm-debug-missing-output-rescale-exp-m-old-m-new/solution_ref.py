import math
import numpy as np


def online_softmax_update(
    m_old,
    l_old,
    O_old,
    m_block,
    l_block,
    O_block,
):
    """Online softmax update computation."""
    if m_old >= m_block:
        m_new = m_old
    else:
        m_new = m_block

    alpha = math.exp(m_old - m_new)
    beta = math.exp(m_block - m_new)

    l_new = alpha * l_old + beta * l_block

    O_old_arr = np.asarray(O_old, dtype=np.float64)
    O_block_arr = np.asarray(O_block, dtype=np.float64)

    c1 = alpha * l_old
    c2 = beta * l_block

    O_new = np.zeros(O_old_arr.shape, dtype=np.float64)
    for i in range(O_old_arr.size):
        O_new.flat[i] = (c1 * O_old_arr.flat[i] + c2 * O_block_arr.flat[i]) / l_new

    return float(m_new), float(l_new), np.asarray(O_new, dtype=np.float64)
