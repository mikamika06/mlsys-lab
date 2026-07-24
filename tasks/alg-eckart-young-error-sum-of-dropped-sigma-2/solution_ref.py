import numpy as np


def eckart_young_errors(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    r = min(X.shape)

    direct = []
    theorem = []
    for k in range(r + 1):
        if k == 0:
            xk = np.zeros_like(X)
        else:
            xk = (u[:, :k] * s[:k]) @ vt[:k, :]
        direct.append(np.sum((X - xk) ** 2))
        theorem.append(np.sum(s[k:] ** 2))

    return np.asarray(direct, dtype=np.float64), np.asarray(theorem, dtype=np.float64)
