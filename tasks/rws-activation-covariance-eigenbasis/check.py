import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    # Generate several random test cases of varying size and dimensionality
    rng = np.random.default_rng(0)
    cases = [
        (rng.standard_normal((5, 3)), "small"),
        (rng.standard_normal((10, 4)), "medium"),
        (rng.standard_normal((20, 6)), "larger"),
        (rng.standard_normal((50, 2)), "wide"),
        (rng.standard_normal((100, 8)), "big")
    ]

    rel_err_max = 0.0
    eigvec_ok = True

    for X, name in cases:
        try:
            C_out, eigvals_out, eigvecs_out = sol.cov_and_eig(X)
        except Exception:
            return {"rel_err": 1.0, "eigvec_match": 0.0}

        # Reference covariance
        n = X.shape[0]
        C_ref = (X.T @ X) / n

        # Reference eigen decomposition
        w_ref, v_ref = np.linalg.eigh(C_ref)
        idx = np.argsort(w_ref)[::-1]          # descending order
        w_sorted = w_ref[idx]
        v_sorted = v_ref[:, idx]

        # Relative error for covariance and eigenvalues
        rel_err_C = scorers.rel_err(C_ref, C_out)
        rel_err_eigvals = scorers.rel_err(w_sorted, eigvals_out)
        rel_err_max = max(rel_err_max, rel_err_C, rel_err_eigvals)

        # Eigenvector consistency (up to sign)
        for j in range(v_sorted.shape[1]):
            v_ref_col = v_sorted[:, j]
            v_out_col = eigvecs_out[:, j]
            diff1 = np.linalg.norm(v_ref_col - v_out_col)
            diff2 = np.linalg.norm(v_ref_col + v_out_col)
            if min(diff1, diff2) > 1e-7:
                eigvec_ok = False
                break

        if not eigvec_ok:
            break

    return {"rel_err": rel_err_max, "eigvec_match": 1.0 if eigvec_ok else 0.0}
