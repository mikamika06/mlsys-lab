import math
import numpy as np

def blockwise_softmax(logits: np.ndarray, block_size: int) -> np.ndarray:
    """Compute softmax using the online blockwise recurrence (flash style).

    Maintains a running maximum *m* and denominator *d* across blocks.
    First pass determines (m_final, d_final); second pass normalises.
    """
    logits = np.asarray(logits, dtype=np.float64)
    n = logits.shape[0]

    m_prev = -float('inf')
    d_prev = 0.0

    for start in range(0, n, block_size):
        block = logits[start:start + block_size]
        m_block = -float('inf')
        for val in block:
            if val > m_block:
                m_block = val
        m_new = m_prev if m_prev > m_block else m_block
        sum_exp = 0.0
        for val in block:
            sum_exp += math.exp(val - m_new)
        d_new = d_prev * math.exp(m_prev - m_new) + sum_exp
        m_prev = m_new
        d_prev = d_new

    m_final = m_prev
    d_final = d_prev
    res = [math.exp(val - m_final) / d_final for val in logits]
    return np.array(res, dtype=np.float64)
