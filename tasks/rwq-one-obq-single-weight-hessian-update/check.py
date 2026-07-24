import numpy as np


def _oracle(w, Hinv, grid):
    w = np.asarray(w, dtype=np.float64)
    Hinv = np.asarray(Hinv, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    nearest = grid[np.argmin(np.abs(w[:, None] - grid[None, :]), axis=1)]
    costs = ((nearest - w) ** 2) / np.diag(Hinv)

    k = int(np.argmin(costs))
    q = nearest[k]
    err = q - w[k]

    out = w.copy()
    out[k] = q
    out -= err * Hinv[:, k] / Hinv[k, k]
    out[k] = q
    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0.8, -1.2, 0.3], dtype=np.float64),
            np.array([[2.0, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 3.0]]),
            np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
        ),
        (
            np.array([0.11, 0.91, -0.77, 0.42], dtype=np.float64),
            np.array([[1.5, 0.1, 0.0, 0.0], [0.1, 2.0, 0.2, 0.0], [0.0, 0.2, 1.2, 0.1], [0.0, 0.0, 0.1, 1.8]]),
            np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
        ),
        (
            np.array([-0.31, 0.62, 1.41], dtype=np.float64),
            np.array([[3.0, 0.2, 0.1], [0.2, 1.4, 0.05], [0.1, 0.05, 2.5]]),
            np.array([-1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0]),
        ),
    ]

    worst = 0.0
    for w, Hinv, grid in cases:
        try:
            got = np.asarray(sol.obq_single_weight_update(w, Hinv, grid), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(w, Hinv, grid)
        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
