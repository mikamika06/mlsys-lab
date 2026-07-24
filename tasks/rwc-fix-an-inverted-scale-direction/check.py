import numpy as np


def _oracle(X, W, s):
    X_new = np.asarray(X, dtype=np.float64) * np.asarray(s, dtype=np.float64)[None, :]
    W_new = np.asarray(W, dtype=np.float64) / np.asarray(s, dtype=np.float64)[:, None]
    return X_new, W_new


def _max_abs_err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[100.0, 1.0, 4.0], [80.0, 2.0, 3.0]]),
            np.array([[0.5, 1.0], [2.0, 0.3], [1.5, 2.5]]),
            np.array([0.01, 3.0, 2.0]),
        ),
        (
            np.array([[6.0, 40.0], [8.0, 30.0], [10.0, 50.0]]),
            np.array([[1.0, 0.2, 0.7], [0.4, 2.0, 1.1]]),
            np.array([0.5, 0.05]),
        ),
        (
            np.array([[3.0, 7.0, 9.0, 2.0]]),
            np.array([[1.0], [2.0], [3.0], [4.0]]),
            np.array([2.0, 0.5, 1.5, 0.25]),
        ),
    ]

    worst_err = 0.0
    range_ok = 1.0
    for X, W, s in cases:
        ref_X, ref_W = _oracle(X, W, s)
        try:
            got_X, got_W = sol.migrate_scale(X.copy(), W.copy(), s.copy())
        except Exception:
            return {"max_abs_err": float("inf"), "range_reduction": 0.0}

        err = max(_max_abs_err(got_X, ref_X), _max_abs_err(got_W, ref_W))
        worst_err = max(worst_err, err)

        wrong_X = X / s[None, :]
        ref_range = float(np.max(np.abs(ref_X)))
        wrong_range = float(np.max(np.abs(wrong_X)))
        if ref_range > wrong_range + 1e-12:
            range_ok = 0.0

    return {
        "max_abs_err": worst_err,
        "range_reduction": range_ok,
    }
