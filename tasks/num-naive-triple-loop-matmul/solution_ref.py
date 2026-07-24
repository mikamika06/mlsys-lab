import numpy as np

def naive_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    m, k1 = A.shape
    k2, n = B.shape
    assert k1 == k2, "Inner dimensions must agree"
    C = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k1):
                s += A[i, p] * B[p, j]
            C[i, j] = s
    return C
