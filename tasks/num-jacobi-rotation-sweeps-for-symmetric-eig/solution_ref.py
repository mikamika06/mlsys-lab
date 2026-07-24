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
        off = np.sqrt(2.0 * off)
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
                    t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = t * c

                Ap = A[:, p].copy()
                Aq = A[:, q].copy()
                A[:, p] = c * Ap - s * Aq
                A[:, q] = s * Ap + c * Aq

                Ap = A[p, :].copy()
                Aq = A[q, :].copy()
                A[p, :] = c * Ap - s * Aq
                A[q, :] = s * Ap + c * Aq
    return np.sort(np.diag(A))
