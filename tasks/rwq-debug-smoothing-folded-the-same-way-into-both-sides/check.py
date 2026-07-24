import numpy as np


def _oracle(X, W, s):
    s = np.asarray(s, dtype=np.float64)
    X_new = np.asarray(X, dtype=np.float64) * s.reshape(1, -1)
    W_new = np.asarray(W, dtype=np.float64) / s.reshape(-1, 1)
    return X_new, W_new


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    X = rng.normal(size=(8, 6)).astype(np.float64)
    W = rng.normal(size=(6, 5)).astype(np.float64)
    s = np.array([0.5, 2.0, 1.5, 3.0, 0.75, 4.0], dtype=np.float64)

    ref_x, ref_w = _oracle(X, W, s)
    ref_product = ref_x @ ref_w

    try:
        got_x, got_w = sol.fold_smoothing(X, W, s)
        got_product = np.asarray(got_x, dtype=np.float64) @ np.asarray(got_w, dtype=np.float64)
        err = float(np.max(np.abs(got_product - ref_product)))
    except Exception:
        err = float("inf")

    return {"max_abs_err": err}
