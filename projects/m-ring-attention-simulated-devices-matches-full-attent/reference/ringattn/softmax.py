import numpy as np


def online_update(m_prev, l_prev, o_prev, scores, v):
    m_curr = np.max(scores, axis=-1, keepdims=True)
    m_new = np.maximum(m_prev, m_curr)
    alpha = np.exp(m_prev - m_new)
    l_new = alpha * l_prev + np.sum(np.exp(scores - m_new), axis=-1, keepdims=True)
    o_new = (alpha * l_prev * o_prev + np.matmul(np.exp(scores - m_new), v)) / l_new
    return m_new, l_new, o_new
