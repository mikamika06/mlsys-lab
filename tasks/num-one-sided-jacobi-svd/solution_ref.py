import numpy as np
import math

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
    
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            val = 0.0
            for k in range(n):
                val += A[k, i] * A[k, j]
            G[i, j] = val

    for _ in range(max_iter):
        best_val = 0.0
        best_i = 0
        best_j = 0
        for i in range(n):
            for j in range(i + 1, n):
                val_abs = G[i, j]
                if val_abs < 0.0:
                    val_abs = -val_abs
                if val_abs > best_val:
                    best_val = val_abs
                    best_i = i
                    best_j = j

        if best_val < tol:
            break

        i, j = best_i, best_j

        tau = (G[j, j] - G[i, i]) / (2.0 * G[i, j])
        tau_abs = tau if tau >= 0.0 else -tau
        if tau_abs < 1e-15:
            t = 1.0
        else:
            sign_tau = 1.0 if tau >= 0.0 else -1.0
            t = sign_tau / (tau_abs + math.sqrt(1.0 + tau ** 2))
        
        c = 1.0 / math.sqrt(1.0 + t ** 2)
        s = t * c

        col_i = np.zeros(n, dtype=np.float64)
        col_j = np.zeros(n, dtype=np.float64)
        for k in range(n):
            col_i[k] = A[k, i]
            col_j[k] = A[k, j]

        for k in range(n):
            A[k, i] = c * col_i[k] - s * col_j[k]
            A[k, j] = s * col_i[k] + c * col_j[k]

        Gi = np.zeros(n, dtype=np.float64)
        Gj = np.zeros(n, dtype=np.float64)
        for k in range(n):
            Gi[k] = G[i, k]
            Gj[k] = G[j, k]

        for k in range(n):
            val = c * Gi[k] - s * Gj[k]
            G[i, k] = val
            G[k, i] = val

        for k in range(n):
            val = s * Gi[k] + c * Gj[k]
            G[j, k] = val
            G[k, j] = val

        G[i, i] = c ** 2 * Gi[i] - 2.0 * s * c * Gi[j] + s ** 2 * Gj[j]
        G[j, j] = s ** 2 * Gi[i] + 2.0 * s * c * Gi[j] + c ** 2 * Gj[j]
        G[i, j] = 0.0
        G[j, i] = 0.0

    raw_sv = np.zeros(n, dtype=np.float64)
    for k in range(n):
        diag_val = G[k, k]
        if diag_val < 0.0:
            diag_val = 0.0
        raw_sv[k] = math.sqrt(diag_val)

    for i in range(n):
        for j in range(i + 1, n):
            if raw_sv[j] > raw_sv[i]:
                tmp = raw_sv[i]
                raw_sv[i] = raw_sv[j]
                raw_sv[j] = tmp

    return raw_sv
