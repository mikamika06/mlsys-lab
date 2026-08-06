import math
import numpy as np


def stream_softmax_row_chunks(logits: np.ndarray, chunk_size: int) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    rows, cols = x.shape
    out = np.empty_like(x, dtype=np.float64)

    for r in range(rows):
        m = -float('inf')
        l = 0.0

        for start in range(0, cols, chunk_size):
            end = min(start + chunk_size, cols)
            mc = -float('inf')
            for c in range(start, end):
                val = x[r, c]
                if val > mc:
                    mc = val
            
            m_new = m if m > mc else mc
            
            term1 = l * math.exp(m - m_new) if m != -float('inf') else 0.0
            term2 = 0.0
            for c in range(start, end):
                term2 += math.exp(x[r, c] - m_new)
            
            l = term1 + term2
            m = m_new

        for start in range(0, cols, chunk_size):
            end = min(start + chunk_size, cols)
            for c in range(start, end):
                out[r, c] = math.exp(x[r, c] - m) / l

    return out
