import math
import numpy as np

def stable_softmax(logits: np.ndarray) -> np.ndarray:
    rows = logits.shape[0]
    cols = logits.shape[1]
    
    max_vals = np.empty((rows, 1), dtype=logits.dtype)
    for i in range(rows):
        m = logits[i, 0]
        for j in range(1, cols):
            if logits[i, j] > m:
                m = logits[i, j]
        max_vals[i, 0] = m

    exp_shifted = np.empty((rows, cols), dtype=logits.dtype)
    for i in range(rows):
        m = max_vals[i, 0]
        for j in range(cols):
            exp_shifted[i, j] = math.exp(logits[i, j] - m)

    sums = np.empty((rows, 1), dtype=logits.dtype)
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += exp_shifted[i, j]
        sums[i, 0] = s

    probs = np.empty((rows, cols), dtype=logits.dtype)
    for i in range(rows):
        s = sums[i, 0]
        for j in range(cols):
            probs[i, j] = exp_shifted[i, j] / s

    return probs
