import numpy as np


def wanda_mask(W, col_norms, keep_ratio):
    W = np.asarray(W, dtype=np.float64)
    col_norms = np.asarray(col_norms, dtype=np.float64)
    rows, cols = W.shape
    k = max(1, int(round(cols * keep_ratio)))

    mask_rows = []
    for i in range(rows):
        row_scores = []
        for j in range(cols):
            val = W[i, j]
            abs_val = val if val >= 0.0 else -val
            score = abs_val * col_norms[j]
            row_scores.append((score, j))
        sorted_row = sorted(row_scores, key=lambda x: (-x[0], x[1]))
        top_indices = [j for _, j in sorted_row[:k]]
        row_mask = [False] * cols
        for j in top_indices:
            row_mask[j] = True
        mask_rows.append(row_mask)
    return np.array(mask_rows, dtype=bool)
