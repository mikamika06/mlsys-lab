import numpy as np
from mlsys import scorers


def _oracle(X):
    X = np.asarray(X, dtype=np.float64)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    r = min(X.shape)
    direct = []
    theorem = []
    for k in range(r + 1):
        if k == 0:
            approx = np.zeros_like(X)
        else:
            approx = (u[:, :k] * s[:k]) @ vt[:k, :]
        direct.append(float(np.sum((X - approx) ** 2)))
        theorem.append(float(np.sum(s[k:] ** 2)))
    return np.asarray(direct), np.asarray(theorem)


def grade(sol, fx) -> dict:
    cases = [
        np.array([[3.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.array([[2.0, -1.0], [0.5, 4.0], [3.0, 2.0]]),
        np.arange(16, dtype=np.float64).reshape(4, 4) / 7.0,
    ]
    all_ref = []
    all_got = []
    for X in cases:
        ref = _oracle(X)
        try:
            got = sol.eckart_young_errors(X)
            if len(got) != 2:
                return {"mse": float("inf")}
            got_arr = np.concatenate([
                np.asarray(got[0], dtype=np.float64).ravel(),
                np.asarray(got[1], dtype=np.float64).ravel(),
            ])
        except Exception:
            return {"mse": float("inf")}
        ref_arr = np.concatenate([ref[0], ref[1]])
        all_ref.append(ref_arr)
        all_got.append(got_arr)
    return {"mse": scorers.mse(np.concatenate(all_ref), np.concatenate(all_got))}
