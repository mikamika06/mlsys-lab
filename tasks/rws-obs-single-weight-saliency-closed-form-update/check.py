import numpy as np


def _oracle(H, w):
    Hinv = np.linalg.inv(H)
    diag = np.diag(Hinv)
    s = w ** 2 / diag
    q = int(np.argmin(s))
    c = w[q] / diag[q]
    delta_w = -c * Hinv[:, q]
    dL = 0.5 * float(delta_w @ H @ delta_w)
    return q, delta_w, dL


def grade(sol, fx) -> dict:
    """
    Builds several seeded random SPD Hessians H and weight vectors w,
    computes the Optimal-Brain-Surgeon single-weight saliency
    s_q = w_q^2 / [H^-1]_qq for every candidate q, the argmin index, its
    closed-form update delta_w = -(w_q/[H^-1]_qq) H^-1 e_q, and the
    resulting analytic second-order loss change
    dL = 0.5 * delta_w^T H delta_w, with a NumPy oracle. Compares the
    submission's chosen index (exact), delta_w (max abs error), and dL
    (relative error) to the oracle's.
    """
    rng = np.random.default_rng(0)
    idx_ok = 1.0
    dw_worst = 0.0
    dL_rel_worst = 0.0
    for _ in range(5):
        d = int(rng.integers(4, 9))
        A = rng.normal(size=(d + 4, d))
        H = A.T @ A + 0.1 * np.eye(d)
        w = rng.normal(size=d)

        q_exp, dw_exp, dL_exp = _oracle(H, w)

        try:
            q_got, dw_got, dL_got = sol.obs_prune_step(H.copy(), w.copy())
            q_got = int(q_got)
            dw_got = np.asarray(dw_got, dtype=np.float64)
            dL_got = float(dL_got)
        except Exception:
            return {"argmin_index_match": 0.0, "deltaw_max_abs_err": float("inf"), "deltaL_rel_err": float("inf")}

        if q_got != q_exp:
            idx_ok = 0.0

        if dw_got.shape != dw_exp.shape:
            dw_worst = float("inf")
        else:
            dw_worst = max(dw_worst, float(np.max(np.abs(dw_got - dw_exp))))

        dL_rel = abs(dL_got - dL_exp) / (abs(dL_exp) + 1e-12)
        dL_rel_worst = max(dL_rel_worst, dL_rel)

    return {
        "argmin_index_match": idx_ok,
        "deltaw_max_abs_err": dw_worst,
        "deltaL_rel_err": dL_rel_worst,
    }
