import numpy as np

from mlsys import scorers


def _bad_matrix(n, seed, tiny):
    """Ill-scaled matrix: A[0, 0] is tiny but nonzero, everything else O(1)."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, n))
    A = rng.uniform(0.5, 2.0, size=(n, n)) * signs
    A[0, 0] = tiny
    return A.tolist()


def _structure_ok(P, U, L, n):
    P = np.asarray(P, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if P.shape != (n, n) or L.shape != (n, n) or U.shape != (n, n):
        return False
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(L)) and np.all(np.isfinite(U))):
        return False
    # L: unit lower triangular
    if not np.allclose(np.triu(L, 1), 0.0, atol=1e-9):
        return False
    if not np.allclose(np.diag(L), 1.0, atol=1e-9):
        return False
    # U: upper triangular
    if not np.allclose(np.tril(U, -1), 0.0, atol=1e-9):
        return False
    # P: a genuine permutation matrix
    if not np.allclose(P @ P.T, np.eye(n), atol=1e-9):
        return False
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)
    if not (np.allclose(row_sums, 1.0, atol=1e-9) and np.allclose(col_sums, 1.0, atol=1e-9)):
        return False
    return True


def grade(sol, fx) -> dict:
    cases = [fx["A"]]
    # extra, independently generated ill-scaled matrices (different size,
    # tiny magnitude and sign) so a fix that only special-cases the fixture
    # values cannot pass.
    cases.append(_bad_matrix(5, seed=11, tiny=3.7e-13))
    cases.append(_bad_matrix(9, seed=42, tiny=-6.1e-12))

    worst_err = 0.0
    for A_list in cases:
        A = np.asarray(A_list, dtype=np.float64)
        n = A.shape[0]
        try:
            out = sol.lu_partial_pivot(A_list)
            P, L, U = out[0], out[1], out[2]
        except Exception:
            return {"max_abs_err": float("inf")}

        if not _structure_ok(P, U, L, n):
            return {"max_abs_err": float("inf")}

        recon_lhs = np.asarray(P, dtype=np.float64) @ A
        recon_rhs = np.asarray(L, dtype=np.float64) @ np.asarray(U, dtype=np.float64)
        err = scorers.max_abs_err(recon_lhs, recon_rhs)
        worst_err = max(worst_err, err)

    return {"max_abs_err": worst_err}
