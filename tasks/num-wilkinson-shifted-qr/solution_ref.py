import numpy as np


def wilkinson_eigvals(A: np.ndarray, max_iter: int = 200) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64).copy()
    m = A.shape[0]
    eps = 1e-12

    for _ in range(max_iter):
        if m <= 1:
            break

        if abs(A[m - 1, m - 2]) <= eps:
            A[m - 1, m - 2] = 0.0
            A[m - 2, m - 1] = 0.0
            m -= 1
            continue

        a = A[m - 2, m - 2]
        b = A[m - 2, m - 1]
        d = A[m - 1, m - 1]

        delta = (a - d) / 2.0
        sign = 1.0 if delta >= 0 else -1.0
        mu = d - (b * b) / (delta + sign * np.sqrt(delta * delta + b * b))

        Q, R = np.linalg.qr(A[:m, :m] - mu * np.eye(m))
        A[:m, :m] = R @ Q + mu * np.eye(m)

    return np.sort(np.diag(A).copy())
