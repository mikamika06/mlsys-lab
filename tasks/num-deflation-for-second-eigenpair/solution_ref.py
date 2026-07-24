import numpy as np


def _power(A, steps=100):
    x = np.ones(A.shape[0], dtype=np.float64)
    x /= np.linalg.norm(x)
    for _ in range(steps):
        x = A @ x
        n = np.linalg.norm(x)
        if n == 0:
            return 0.0, np.zeros_like(x)
        x /= n
    value = float(x @ (A @ x))
    return value, x


def second_eigenvalue(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.float64)
    lam1, v1 = _power(A)
    B = A - lam1 * np.outer(v1, v1)
    lam2, _ = _power(B)
    return float(lam2)
