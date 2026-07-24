import numpy as np


def shifted_inverse_iteration(A, mu, x0, iters):
    A = np.asarray(A, dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64).copy()

    x = x / np.linalg.norm(x)
    shifted = A - mu * np.eye(A.shape[0], dtype=np.float64)

    for _ in range(iters):
        y = np.linalg.solve(shifted, x)
        x = y / np.linalg.norm(y)

    return float((x @ A @ x) / (x @ x))
