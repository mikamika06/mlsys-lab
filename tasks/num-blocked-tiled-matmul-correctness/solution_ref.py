import numpy as np


def blocked_matmul(A: np.ndarray, B: np.ndarray, block_size: int) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    m, k = A.shape
    _, n = B.shape
    C = np.zeros((m, n), dtype=np.float64)

    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            for kk in range(0, k, block_size):
                i_end = min(i + block_size, m)
                j_end = min(j + block_size, n)
                k_end = min(kk + block_size, k)
                C[i:i_end, j:j_end] += (
                    A[i:i_end, kk:k_end] @ B[kk:k_end, j:j_end]
                )

    return C
