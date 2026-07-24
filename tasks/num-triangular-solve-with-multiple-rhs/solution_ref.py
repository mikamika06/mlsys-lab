import numpy as np


def solve_lower_multi_rhs(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = L.shape[0]
    X = np.zeros_like(B, dtype=np.float64)

    for i in range(n):
        X[i] = (B[i] - L[i, :i] @ X[:i]) / L[i, i]

    return X
