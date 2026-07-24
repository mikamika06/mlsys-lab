import numpy as np


def batched_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    batch, m, k = A.shape
    _, _, n = B.shape
    out = np.zeros((batch, m, n), dtype=np.float64)

    for s in range(batch):
        for i in range(m):
            for j in range(n):
                total = 0.0
                for t in range(k):
                    total += A[s, i, t] * B[s, t, j]
                out[s, i, j] = total

    return out
