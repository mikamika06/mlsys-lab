import numpy as np

from mlsys import scorers


def _oracle(X, damp_pct):
    d = X.shape[1]
    H = 2.0 * (X.T @ X)
    diag_mean = float(np.mean(np.diag(H)))
    damp = damp_pct * diag_mean
    H_damped = H + damp * np.eye(d)
    Hinv = np.linalg.inv(H_damped)
    L = np.linalg.cholesky(Hinv)
    U = L.T
    return H_damped, Hinv, U


def _synthetic_cases():
    rng = np.random.default_rng(107)
    cases = []
    for _ in range(4):
        n_cal = int(rng.integers(40, 120))
        d_in = int(rng.integers(4, 16))
        mix = rng.standard_normal((d_in, d_in)) / np.sqrt(d_in)
        X = rng.standard_normal((n_cal, d_in)) @ mix
        damp_pct = float(rng.uniform(0.005, 0.05))
        cases.append((X, damp_pct))
    return cases


def grade(sol, fx) -> dict:
    cases = [(fx["X"], 0.01)] + _synthetic_cases()

    worst = 0.0
    for X, damp_pct in cases:
        ref_H, ref_Hinv, ref_U = _oracle(X, damp_pct)
        try:
            got = sol.damped_inv_hessian_cholesky(X.copy(), damp_pct)
            got_H = np.asarray(got["H"], dtype=np.float64)
            got_Hinv = np.asarray(got["Hinv"], dtype=np.float64)
            got_U = np.asarray(got["U"], dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}

        if got_H.shape != ref_H.shape or got_Hinv.shape != ref_Hinv.shape or got_U.shape != ref_U.shape:
            return {"rel_err": float("inf")}

        ref_vec = np.concatenate([ref_H.ravel(), ref_Hinv.ravel(), ref_U.ravel()])
        got_vec = np.concatenate([got_H.ravel(), got_Hinv.ravel(), got_U.ravel()])
        err = scorers.rel_err(ref_vec, got_vec)
        worst = max(worst, err)

    return {"rel_err": worst}
