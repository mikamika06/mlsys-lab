import numpy as np


def _oracle(U, S, Vt, X_new, k):
    current = U @ np.diag(S) @ Vt
    full = np.vstack([current, X_new])
    _, s, _ = np.linalg.svd(full, full_matrices=False)
    return s[:k]


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = []
    for m, n, old_rank, rows, k in [
        (6, 4, 3, 2, 2),
        (8, 5, 4, 3, 3),
        (5, 3, 2, 4, 2),
    ]:
        U0, _ = np.linalg.qr(rng.normal(size=(m, old_rank)))
        V0, _ = np.linalg.qr(rng.normal(size=(n, old_rank)))
        S0 = np.sort(rng.uniform(0.5, 5.0, size=old_rank))[::-1]
        X = rng.normal(size=(rows, n))
        cases.append((U0, S0, V0.T, X, k))

    errs = []
    for U, S, Vt, X, k in cases:
        try:
            out = sol.incremental_svd_update(U, S, Vt, X, k)
            s = np.asarray(out[1], dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _oracle(U, S, Vt, X, k)
        errs.append(np.linalg.norm(s - ref) / (np.linalg.norm(ref) + 1e-12))

    return {"rel_err": float(max(errs))}
