import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    # Generate ill‑conditioned test cases
    rng = np.random.default_rng(42)
    tests = []

    # Case 1: two columns almost perfectly correlated
    X1 = rng.standard_normal((50, 5))
    X1[:, 1] = X1[:, 0] * 2.0 + rng.normal(scale=1e-4, size=50)
    tests.append((X1, 3))

    # Case 2: three columns nearly linear combinations
    X2 = rng.standard_normal((80, 8))
    for i in range(1, 8):
        X2[:, i] = X2[:, 0] * (i + 1) + rng.normal(scale=1e-4, size=80)
    tests.append((X2, 5))

    # Case 3: small scale on some columns
    X3 = rng.standard_normal((200, 10))
    X3[:, 5:] *= 1e-6
    tests.append((X3, 7))

    max_err = 0.0

    for X, k in tests:
        try:
            eig_vecs, svd_vecs = sol.cov_eig_vs_svd_pca(X, k)
        except Exception:
            return {"max_abs_err": float("inf")}

        # Reference via SVD
        Xc = X - np.mean(X, axis=0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        ref_vecs = Vt[:k]

        # Validate shapes and dtype
        if eig_vecs.shape != ref_vecs.shape or eig_vecs.dtype != np.float64:
            return {"max_abs_err": float("inf")}

        # Align signs (assume ordering matches)
        for i in range(k):
            if np.dot(eig_vecs[i], ref_vecs[i]) < 0:
                eig_vecs[i] *= -1

        err = max_abs_err(ref_vecs, eig_vecs)
        if err > max_err:
            max_err = err
        if max_err > 1e-6:
            break

    return {"max_abs_err": max_err}
