import numpy as np

_EPS = np.finfo(np.float64).eps


def _offdiag_norm(A: np.ndarray) -> float:
    n = A.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.sqrt(np.sum(A[mask] ** 2)))


def _jacobi_sweep(A: np.ndarray) -> np.ndarray:
    """One classical cyclic Jacobi sweep: visit every pivot pair (p, q)
    with p < q, in row-major nested order, applying one rotation each."""
    A = A.copy()
    n = A.shape[0]
    for p in range(n - 1):
        for q in range(p + 1, n):
            apq = A[p, q]
            # Already negligible relative to the diagonal scale: skip.
            # (Prevents dividing by a residual near-zero entry left over
            # from earlier rotations in this or prior sweeps.)
            if abs(apq) <= _EPS * (abs(A[p, p]) + abs(A[q, q])):
                A[p, q] = 0.0
                A[q, p] = 0.0
                continue

            theta = (A[q, q] - A[p, p]) / (2.0 * apq)
            if theta == 0.0:
                t = 1.0
            else:
                t = np.sign(theta) / (abs(theta) + np.sqrt(theta * theta + 1.0))
            c = 1.0 / np.sqrt(t * t + 1.0)
            s = t * c

            app = A[p, p] - t * apq
            aqq = A[q, q] + t * apq
            A[p, p] = app
            A[q, q] = aqq
            A[p, q] = 0.0
            A[q, p] = 0.0

            for i in range(n):
                if i == p or i == q:
                    continue
                aip = A[i, p]
                aiq = A[i, q]
                new_ip = c * aip - s * aiq
                new_iq = s * aip + c * aiq
                A[i, p] = new_ip
                A[p, i] = new_ip
                A[i, q] = new_iq
                A[q, i] = new_iq
    return A


def jacobi_offdiag_norms(A: np.ndarray, n_sweeps: int) -> np.ndarray:
    """
    Run ``n_sweeps`` classical cyclic Jacobi sweeps on the symmetric matrix
    ``A`` and return the off-diagonal Frobenius norm before any sweeps and
    after each sweep, as a 1-D array of length ``n_sweeps + 1``.

    Each sweep visits pivot pairs ``(p, q)`` with ``0 <= p < q < n``, in
    row-major nested order (``p`` outer, ``q`` inner), applying exactly one
    Jacobi rotation per pair using the numerically stable formula:

        theta = (A[q,q] - A[p,p]) / (2*A[p,q])
        t     = 1                                          if theta == 0
              = sign(theta) / (|theta| + sqrt(theta**2+1))  otherwise
        c     = 1 / sqrt(t**2 + 1)
        s     = t * c

    ``A`` itself must not be mutated (work on a copy).
    """
    A = np.asarray(A, dtype=np.float64).copy()
    norms = [_offdiag_norm(A)]
    for _ in range(n_sweeps):
        A = _jacobi_sweep(A)
        norms.append(_offdiag_norm(A))
    return np.array(norms, dtype=np.float64)
