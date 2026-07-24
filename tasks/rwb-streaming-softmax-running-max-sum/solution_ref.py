import numpy as np


def streaming_softmax(scores: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Compute softmax(scores) by streaming through it in chunks of at most
    `chunk_size` elements, maintaining a running max `m` and running sum
    `l` (rescaling `l` whenever a new chunk raises the max), then
    normalizing every element with the final (m, l).
    """
    x = np.asarray(scores, dtype=np.float64)
    n = x.shape[0]

    m = -np.inf
    l = 0.0
    for start in range(0, n, chunk_size):
        chunk = x[start : start + chunk_size]
        chunk_max = chunk.max()
        m_new = max(m, chunk_max)
        alpha = 0.0 if np.isneginf(m) else np.exp(m - m_new)
        l = l * alpha + np.exp(chunk - m_new).sum()
        m = m_new

    return np.exp(x - m) / l
