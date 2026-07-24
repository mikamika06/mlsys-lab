import numpy as np


def randomized_svd(A: np.ndarray, k: int, seed: int):
    m, n = A.shape
    rng = np.random.default_rng(seed)

    p = min(5, n - k)
    q = k + p

    omega = rng.normal(size=(n, q))
    Y = A @ omega

    Q, _ = np.linalg.qr(Y, mode="reduced")

    B = Q.T @ A
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)

    return Q @ Ub[:, :k], S[:k], Vt[:k, :]
