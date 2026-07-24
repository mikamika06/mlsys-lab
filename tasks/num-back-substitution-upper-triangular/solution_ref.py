import numpy as np

def back_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = U.shape[0]
    x = np.empty(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if i < n - 1:
            s = U[i, i + 1:] @ x[i + 1:]
        else:
            s = 0.0
        x[i] = (b[i] - s) / U[i, i]
    return x
