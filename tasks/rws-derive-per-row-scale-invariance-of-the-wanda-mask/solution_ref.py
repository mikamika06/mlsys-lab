import numpy as np


def wanda_mask(W: np.ndarray, col_norms: np.ndarray, keep_ratio: float) -> np.ndarray:
    rows = W.shape[0]
    cols = W.shape[1]
    k = max(1, int(round(cols * keep_ratio)))

    mask = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        row_data = []
        for j in range(cols):
            val = W[i, j]
            abs_val = val if val >= 0.0 else -val
            score = abs_val * col_norms[j]
            row_data.append((j, score))
        
        sorted_row = sorted(row_data, key=lambda x: -x[1])
        for idx in range(k):
            j_idx = sorted_row[idx][0]
            mask[i, j_idx] = True

    return mask
