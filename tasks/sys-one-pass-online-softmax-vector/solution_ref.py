import math
import numpy as np


def online_softmax_weighted_sum(scores: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    """softmax(scores) @ V, computed one block at a time via the online-softmax
    running (m, l, o) update -- never calling exp on the full-length score
    vector. See task.md for the update rule.
    """
    scores = np.asarray(scores, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n = scores.shape[0]
    d = V.shape[1]

    m = -float('inf')
    l = 0.0
    o = np.zeros(d, dtype=np.float64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        
        m_block = -float('inf')
        for i in range(start, end):
            val = float(scores[i])
            if val > m_block:
                m_block = val

        m_new = m if m > m_block else m_block

        if math.isfinite(m):
            correction = math.exp(m - m_new)
        else:
            correction = 0.0

        sum_p = 0.0
        p_list = []
        for i in range(start, end):
            val = math.exp(float(scores[i]) - m_new)
            p_list.append(val)
            sum_p += val

        l = l * correction + sum_p

        for j in range(d):
            o[j] = o[j] * correction

        for i_idx, i in enumerate(range(start, end)):
            p_val = p_list[i_idx]
            for j in range(d):
                o[j] += p_val * float(V[i, j])

        m = m_new

    result = np.empty(d, dtype=np.float64)
    for j in range(d):
        result[j] = o[j] / l

    return result
