import numpy as np


def _reference(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X_norm = (X**2).sum(axis=1)[:, None]
    Y_norm = (Y**2).sum(axis=1)[None, :]
    return X_norm + Y_norm - 2 * X.dot(Y.T)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (5, 3, 4),
        (10, 8, 6),
        (20, 12, 15),
    ]
    mse_total = 0.0
    for n, d, m in cases:
        X = rng.standard_normal((n, d)).tolist()
        Y = rng.standard_normal((m, d)).tolist()
        try:
            got = sol.pairwise_l2_matrix(X, Y)
        except Exception:
            return {"mse": float("inf")}
        got_arr = np.asarray(got, dtype=np.float64)
        ref = _reference(X, Y)
        mse_total += np.mean((got_arr - ref) ** 2)
    mse_avg = mse_total / len(cases)
    return {"mse": mse_avg}
