import math
import numpy as np

def softmax_stats(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per‑row maximum and sum of exponentials after shifting by that maximum.
    Uses only O(K) temporary memory per row.
    """
    n_rows = logits.shape[0]
    row_max = np.empty(n_rows, dtype=np.float64)
    row_sum_exp = np.empty(n_rows, dtype=np.float64)

    for i in range(n_rows):
        row = logits[i]
        n_cols = row.shape[0]
        
        m = row[0]
        for j in range(1, n_cols):
            val = row[j]
            if val > m:
                m = val
        row_max[i] = m

        s = 0.0
        for j in range(n_cols):
            s += math.exp(row[j] - m)
        row_sum_exp[i] = s

    return row_max, row_sum_exp
