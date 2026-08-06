import numpy as np


def sum_rows_fp32(A: np.ndarray) -> np.ndarray:
    rows = A.shape[0]
    cols = A.shape[1]
    result = np.zeros(rows, dtype=np.float32)
    for i in range(rows):
        acc = 0.0
        for j in range(cols):
            acc += float(A[i, j])
        result[i] = np.float32(acc)
    return result
