import numpy as np

def one_sided_jacobi_svd(A, tol=1e-12, max_iter=1000):
    """
    Compute singular values of A via the one-sided Jacobi SVD algorithm.

    Parameters
    ----------
    A : np.ndarray of shape (n, n), square real matrix.
    tol : float, convergence tolerance on off-diagonal Frobenius norm.
    max_iter : int, maximum number of Jacobi sweeps.

    Returns
    -------
    singular_values : np.ndarray of shape (n,), sorted in descending order.
    """
    A = A.astype(np.float64).copy()
    n = A.shape[0]
    G = A.T @ A

    for _ in range(max_iter):
        best_val = 0.0
        best_i = best_j = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(G[i, j]) > best_val:
                    best_val = abs(G[i, j])
                    best_i, best_j = i, j

        if best_val < tol:
            break

        i, j = best_i, best_j

        tau = (G[j, j] - G[i, i]) / (2.0 * G[i, j])
        if abs(tau) < 1e-15:
            t = 1.0
        else:
            t = np.sign(tau) / (abs(tau) + np.sqrt(1.0 + tau ** 2))
        c = 1.0 / np.sqrt(1.0 + t ** 2)
        s = t * c

        col_i = A[:, i].copy()
        col_j = A[:, j].copy()
        A[:, i] = c * col_i - s * col_j
        A[:, j] = s * col_i + c * col_j

        Gi = G[i, :].copy()
        Gj = G[j, :].copy()

        G[i, :] = c * Gi - s * Gj
        G[:, i] = G[i, :]
        G[j, :] = s * Gi + c * Gj
        G[:, j] = G[j, :]

        G[i, i] = c ** 2 * Gi[i] - 2.0 * s * c * Gi[j] + s ** 2 * Gj[j]
        G[j, j] = s ** 2 * Gi[i] + 2.0 * s * c * Gi[j] + c ** 2 * Gj[j]
        G[i, j] = 0.0
        G[j, i] = 0.0

    return np.sort(np.sqrt(np.maximum(np.diag(G), 0.0)))[::-1]
