import math
import numpy as np


def jacobi_eigenvalues(A: np.ndarray, sweeps: int = 100, tol: float = 1e-12) -> np.ndarray:
    """Eigenvalues of symmetric A via cyclic Jacobi rotation sweeps, sorted ascending."""
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    for _ in range(sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                off += A[p, q] ** 2
        off = math.sqrt(2.0 * off)
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if abs(apq) < 1e-300:
                    continue
                app, aqq = A[p, p], A[q, q]
                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                Ap = np.zeros(n, dtype=np.float64)
                Aq = np.zeros(n, dtype=np.float64)
                for i in range(n):
                    Ap[i] = A[i, p]
                    Aq[i] = A[i, q]
                for i in range(n):
                    A[i, p] = c * Ap[i] - s * Aq[i]
                    A[i, q] = s * Ap[i] + c * Aq[i]

                Ap_row = np.zeros(n, dtype=np.float64)
                Aq_row = np.zeros(n, dtype=np.float64)
                for j in range(n):
                    Ap_row[j] = A[p, j]
                    Aq_row[j] = A[q, j]
                for j in range(n):
                    A[p, j] = c * Ap_row[j] - s * Aq_row[j]
                    A[q, j] = s * Ap_row[j] + c * Aq_row[j]

    diag_vals = []
    for i in range(n):
        diag_vals.append(A[i, i])
    
    for i in range(len(diag_vals)):
        for j in range(i + 1, len(diag_vals)):
            if diag_vals[i] > diag_vals[j]:
                temp = diag_vals[i]
                diag_vals[i] = diag_vals[j]
                diag_vals[j] = temp

    return np.array(diag_vals, dtype=np.float64)
