import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    n, d = 5, 4
    X = rng.standard_normal((n, d), dtype=np.float64)
    lam = 0.1

    try:
        H, H_inv = sol.hessian_and_inverse(X, lam)
    except Exception:
        return {"max_abs_err": float("inf")}

    H_ref = 2 * X @ X.T + lam * np.eye(n, dtype=np.float64)
    H_inv_ref = np.linalg.inv(H_ref)

    err_H = np.max(np.abs(H - H_ref))
    err_Hinv = np.max(np.abs(H_inv - H_inv_ref))

    return {"max_abs_err": float(max(err_H, err_Hinv))}
