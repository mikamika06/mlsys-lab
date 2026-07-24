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
        m = row.max()
        row_max[i] = m
        exp_shifted = np.exp(row - m)
        row_sum_exp[i] = exp_shifted.sum()

    return row_max, row_sum_exp
