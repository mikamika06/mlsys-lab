import numpy as np

from mlsys import scorers


def _ref_obq_step(W, H_inv, col, scale, nmax):
    W = np.asarray(W, dtype=np.float64).copy()
    w_col = W[:, col]
    codes = np.clip(np.round(w_col / scale), -nmax, nmax)
    q_col = codes * scale
    err = (w_col - q_col) / H_inv[col, col]
    W[:, col] = q_col
    n = W.shape[1]
    if col + 1 < n:
        W[:, col + 1:] -= np.outer(err, H_inv[col, col + 1:])
    return q_col, W


def _make_h_inv(rng, n):
    A = rng.normal(size=(n, n))
    H = A.T @ A + 0.1 * np.eye(n)
    return np.linalg.inv(H)


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []

    for rows, n in [(4, 6), (3, 3), (5, 8)]:
        W = rng.normal(size=(rows, n))
        H_inv = _make_h_inv(rng, n)
        scale = np.abs(rng.normal(size=rows)) + 0.1
        nmax = int(rng.choice([3, 7, 15]))
        for col in [0, n // 2, n - 1]:
            scenarios.append((W, H_inv, col, scale, nmax))

    return scenarios


def grade(sol, fx) -> dict:
    worst = 0.0
    for W, H_inv, col, scale, nmax in _scenarios():
        q_ref, Wu_ref = _ref_obq_step(W, H_inv, col, scale, nmax)

        try:
            q_got, Wu_got = sol.obq_column_step(W.copy(), H_inv.copy(), col, scale.copy(), nmax)
        except Exception:
            return {"rel_err": float("inf")}

        try:
            q_got = np.asarray(q_got, dtype=np.float64)
            Wu_got = np.asarray(Wu_got, dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}

        if q_got.shape != q_ref.shape or Wu_got.shape != Wu_ref.shape:
            return {"rel_err": float("inf")}

        err = max(
            scorers.rel_err(q_ref, q_got),
            scorers.rel_err(Wu_ref, Wu_got),
        )
        if not np.isfinite(err):
            return {"rel_err": float("inf")}
        worst = max(worst, err)

    return {"rel_err": worst}
