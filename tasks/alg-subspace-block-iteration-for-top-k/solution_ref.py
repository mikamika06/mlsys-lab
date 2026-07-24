import numpy as np


def block_power_topk(A: np.ndarray, Q0: np.ndarray, n_iter: int):
    """Subspace (block power) iteration for the top-k eigenpairs of symmetric A."""
    A = np.asarray(A, dtype=np.float64)
    Q, _ = np.linalg.qr(np.asarray(Q0, dtype=np.float64))

    for _ in range(n_iter):
        Z = A @ Q
        Q, _ = np.linalg.qr(Z)

    # Rayleigh-Ritz on the converged subspace.
    M = Q.T @ A @ Q
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    order = np.argsort(w)[::-1]
    eigvals = w[order]
    Q = Q @ V[:, order]
    return eigvals.astype(np.float64), np.ascontiguousarray(Q, dtype=np.float64)
