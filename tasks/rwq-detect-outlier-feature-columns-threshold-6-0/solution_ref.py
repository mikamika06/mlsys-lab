import numpy as np


def detect_outlier_columns(X: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    """
    Return the sorted, unique column indices j where max_i |X[i, j]| >= threshold.
    """
    X = np.asarray(X, dtype=np.float64)
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    idx = []
    for j in range(n_cols):
        col_max = 0.0
        for i in range(n_rows):
            val = X[i, j]
            if val < 0.0:
                val = -val
            if val > col_max:
                col_max = val
        if col_max >= threshold:
            idx.append(j)
    return np.array(idx, dtype=np.int64)
