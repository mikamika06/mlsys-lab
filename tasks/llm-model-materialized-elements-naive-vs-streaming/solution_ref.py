import math
import numpy as np

def softmax_streaming(logits: np.ndarray) -> np.ndarray:
    """Stable vectorized softmax applied row‑wise."""
    rows, cols = logits.shape
    out = np.empty((rows, cols), dtype=logits.dtype)
    for i in range(rows):
        max_val = logits[i, 0]
        for j in range(1, cols):
            if logits[i, j] > max_val:
                max_val = logits[i, j]

        row_sum = 0.0
        for j in range(cols):
            val = math.exp(logits[i, j] - max_val)
            out[i, j] = val
            row_sum += val

        for j in range(cols):
            out[i, j] /= row_sum

    return out
