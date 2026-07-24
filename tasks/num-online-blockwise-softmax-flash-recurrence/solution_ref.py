import numpy as np

def blockwise_softmax(logits: np.ndarray, block_size: int) -> np.ndarray:
    """Compute softmax using the online blockwise recurrence (flash style).

    Maintains a running maximum *m* and denominator *d* across blocks.
    First pass determines (m_final, d_final); second pass normalises.
    """
    logits = np.asarray(logits, dtype=np.float64)
    n = logits.shape[0]

    # --- Pass 1: accumulate global m and d ---
    m_prev = -np.inf
    d_prev = 0.0

    for start in range(0, n, block_size):
        block = logits[start:start + block_size]
        m_block = np.max(block)
        m_new = m_prev if m_prev > m_block else m_block
        sum_exp = np.sum(np.exp(block - m_new))
        d_new = d_prev * np.exp(m_prev - m_new) + sum_exp
        m_prev = m_new
        d_prev = d_new

    # --- Pass 2: normalise ---
    m_final = m_prev
    d_final = d_prev
    return np.exp(logits - m_final) / d_final
