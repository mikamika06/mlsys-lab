import numpy as np


def stream_softmax_row_chunks(logits: np.ndarray, chunk_size: int) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    rows, cols = x.shape
    out = np.empty_like(x, dtype=np.float64)

    for r in range(rows):
        m = -np.inf
        l = 0.0

        for start in range(0, cols, chunk_size):
            chunk = x[r, start:start + chunk_size]
            mc = np.max(chunk)
            m_new = max(m, mc)
            l = l * np.exp(m - m_new) + np.sum(np.exp(chunk - m_new))
            m = m_new

        for start in range(0, cols, chunk_size):
            chunk = x[r, start:start + chunk_size]
            out[r, start:start + chunk_size] = np.exp(chunk - m) / l

    return out
