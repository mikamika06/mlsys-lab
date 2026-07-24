import numpy as np

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Naive matrix multiplication using a triple loop."""
    m, p = A.shape
    p2, n = B.shape
    assert p == p2, "Incompatible shapes"
    C = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C
