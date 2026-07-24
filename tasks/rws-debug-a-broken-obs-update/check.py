import numpy as np
from mlsys import scorers


def _ref_obs_update(W, Hinv, q):
    W_out = np.array(W, dtype=np.float64, copy=True)
    column = W_out[:, q].copy()
    denom = Hinv[q, q]
    for j in range(W_out.shape[1]):
        if j != q:
            W_out[:, j] -= column * Hinv[q, j] / denom
    W_out[:, q] = 0.0
    return W_out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.5, -2.0, 0.5], [3.0, 1.0, -1.5]], dtype=np.float64),
            np.array([[2.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 1.2]], dtype=np.float64),
            1,
        ),
        (
            np.array([[4.0, -1.0], [2.0, 3.0], [-2.0, 5.0]], dtype=np.float64),
            np.array([[1.0, 0.4], [0.4, 2.0]], dtype=np.float64),
            0,
        ),
        (
            np.array([[0.5, 2.5, -3.0, 1.0]], dtype=np.float64),
            np.array(
                [
                    [1.8, 0.2, 0.1, 0.0],
                    [0.2, 1.4, 0.25, 0.1],
                    [0.1, 0.25, 2.0, 0.3],
                    [0.0, 0.1, 0.3, 1.1],
                ],
                dtype=np.float64,
            ),
            2,
        ),
    ]

    worst = 0.0
    for W, Hinv, q in cases:
        try:
            got = sol.obs_update(W.copy(), Hinv.copy(), q)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _ref_obs_update(W, Hinv, q)
        worst = max(worst, scorers.rel_err(ref, got))
    return {"rel_err": worst}
