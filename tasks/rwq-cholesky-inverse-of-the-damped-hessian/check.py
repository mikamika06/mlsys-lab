import numpy as np
from mlsys import scorers


def _oracle_inverse(H):
    n = H.shape[0]
    L = np.linalg.cholesky(H)
    L_inv = np.linalg.solve(L, np.eye(n))
    return L_inv.T @ L_inv


def grade(sol, fx) -> dict:
    """
    Random damped Hessians H = X^T X + lambda*I of varying size; compares
    the submitted inverse against a Cholesky-factor NumPy oracle.
    """
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(6):
        n = int(rng.integers(2, 30))
        rows = int(rng.integers(n, n + 20))
        X = rng.standard_normal((rows, n))
        lam = float(rng.uniform(0.01, 1.0))
        H = X.T @ X + lam * np.eye(n)

        expected = _oracle_inverse(H)
        try:
            got = np.asarray(sol.cholesky_inverse(H.copy()), dtype=np.float64)
            if got.shape != expected.shape:
                return {"rel_err": float("inf")}
            err = scorers.rel_err(expected, got)
        except Exception:
            return {"rel_err": float("inf")}

        worst = max(worst, err)

    return {"rel_err": worst}
