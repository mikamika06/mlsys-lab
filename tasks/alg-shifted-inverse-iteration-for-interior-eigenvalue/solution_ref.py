import numpy as np


def shifted_inverse_iteration(A, sigma, iterations):
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    x = np.ones(n, dtype=np.float64)
    x /= np.linalg.norm(x)

    shifted = A - sigma * np.eye(n, dtype=np.float64)
    for _ in range(iterations):
        y = np.linalg.solve(shifted, x)
        x = y / np.linalg.norm(y)

    value = float((x @ A @ x) / (x @ x))
    return value, x
