import numpy as np


def int8_gemv(A: np.ndarray, x: np.ndarray):
    """Reference: row-major traversal, deterministic access trace."""
    m, n = A.shape
    y = np.zeros(m, dtype=np.int32)
    access = []
    base_A = 0
    base_x = m * n
    for i in range(m):
        acc = 0
        for j in range(n):
            access.append(base_A + i * n + j)
            access.append(base_x + j)
            acc += int(A[i, j]) * int(x[j])
        y[i] = acc
    return y, access
