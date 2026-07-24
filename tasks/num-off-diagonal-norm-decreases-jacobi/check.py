import numpy as np

from mlsys import scorers

N_SWEEPS = 8
_EPS = np.finfo(np.float64).eps


def _offdiag_norm(A):
    n = A.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.sqrt(np.sum(A[mask] ** 2)))


def _jacobi_sweep(A):
    A = A.copy()
    n = A.shape[0]
    for p in range(n - 1):
        for q in range(p + 1, n):
            apq = A[p, q]
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


def _oracle_norms(A, n_sweeps):
    """Real oracle: an independent, direct implementation of the exact
    classical cyclic Jacobi sweep algorithm specified in task.md."""
    A = np.asarray(A, dtype=np.float64).copy()
    norms = [_offdiag_norm(A)]
    for _ in range(n_sweeps):
        A = _jacobi_sweep(A)
        norms.append(_offdiag_norm(A))
    return np.array(norms, dtype=np.float64)


def _second_case():
    """A second, independently generated symmetric matrix so the sequence
    cannot be memorised from the fixture alone."""
    rng = np.random.default_rng(13)
    n = 8
    B = rng.standard_normal((n, n))
    return 0.5 * (B + B.T)


def grade(sol, fx) -> dict:
    cases = [np.asarray(fx["A"], dtype=np.float64), _second_case()]

    worst_err = 0.0
    for A in cases:
        ref = _oracle_norms(A, N_SWEEPS)
        try:
            got = sol.jacobi_offdiag_norms(A, N_SWEEPS)
            got = np.asarray(got, dtype=np.float64).ravel()
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}
        # off-diagonal norm can never be negative
        if np.any(got < -1e-12):
            return {"max_abs_err": float("inf")}
        # must be (numerically) monotonically non-increasing
        if np.any(np.diff(got) > 1e-9):
            return {"max_abs_err": float("inf")}
        # must have converged by the last sweep
        if got[-1] >= 1e-9:
            return {"max_abs_err": float("inf")}

        worst_err = max(worst_err, scorers.max_abs_err(ref, got))

    return {"max_abs_err": worst_err}
