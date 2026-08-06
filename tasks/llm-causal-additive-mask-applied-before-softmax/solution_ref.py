import math
import numpy as np

def causal_masked_softmax(scores: np.ndarray) -> np.ndarray:
    n_rows, n_cols = scores.shape
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        row_sum = 0.0
        row_exps = []
        for j in range(n_cols):
            if j <= i:
                val = math.exp(scores[i, j])
            else:
                val = math.exp(-float('inf'))
            row_exps.append(val)
            row_sum += val
        for j in range(n_cols):
            out[i, j] = row_exps[j] / row_sum
    return out
