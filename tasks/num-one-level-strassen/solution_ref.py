import numpy as np


def one_level_strassen(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    m = n // 2

    A11, A12 = A[:m, :m], A[:m, m:]
    A21, A22 = A[m:, :m], A[m:, m:]
    B11, B12 = B[:m, :m], B[:m, m:]
    B21, B22 = B[m:, :m], B[m:, m:]

    M1 = (A11 + A22) @ (B11 + B22)
    M2 = (A21 + A22) @ B11
    M3 = A11 @ (B12 - B22)
    M4 = A22 @ (B21 - B11)
    M5 = (A11 + A12) @ B22
    M6 = (A21 - A11) @ (B11 + B12)
    M7 = (A12 - A22) @ (B21 + B22)

    C = np.empty((n, n), dtype=np.float64)
    C[:m, :m] = M1 + M4 - M5 + M7
    C[:m, m:] = M3 + M5
    C[m:, :m] = M2 + M4
    C[m:, m:] = M1 - M2 + M3 + M6
    return C
