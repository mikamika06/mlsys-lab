import numpy as np


def sum_rows_fp32(A: np.ndarray) -> np.ndarray:
    return np.sum(A, axis=1, dtype=np.float32)
