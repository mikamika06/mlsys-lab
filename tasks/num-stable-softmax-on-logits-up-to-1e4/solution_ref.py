import math
import numpy as np

def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    rows, cols = logits.shape
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        max_val = logits[i, 0]
        for j in range(1, cols):
            if logits[i, j] > max_val:
                max_val = logits[i, j]
        sum_val = 0.0
        for j in range(cols):
            exp_val = math.exp(logits[i, j] - max_val)
            out[i, j] = exp_val
            sum_val += exp_val
        for j in range(cols):
            out[i, j] /= sum_val
    return out
