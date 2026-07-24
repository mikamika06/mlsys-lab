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

    m = -np.inf
    l = 0.0
    o = np.zeros(d, dtype=np.float64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        chunk = scores[start:end]
        v_chunk = V[start:end]

        m_block = float(np.max(chunk))
        m_new = max(m, m_block)

        correction = np.exp(m - m_new) if np.isfinite(m) else 0.0
        p = np.exp(chunk - m_new)

        l = l * correction + float(np.sum(p))
        o = o * correction + p @ v_chunk
        m = m_new

    return o / l
