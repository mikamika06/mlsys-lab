import numpy as np


def _oracle(W, b, x, keep_rows, keep_cols):
    Wp = np.asarray(W, dtype=np.float64)[keep_rows][:, keep_cols]
    bp = np.asarray(b, dtype=np.float64)[keep_rows]
    xp = np.asarray(x, dtype=np.float64)[keep_cols]
    return Wp @ xp + bp


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]),
            np.array([0.5, -1.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([1]),
            np.array([0, 2]),
        ),
        (
            np.arange(30, dtype=np.float64).reshape(5, 6) / 3.0,
            np.linspace(-1.0, 1.0, 5),
            np.array([1.5, -2.0, 0.5, 3.0, 4.0, -1.0]),
            np.array([0, 2, 4]),
            np.array([1, 3, 5]),
        ),
        (
            np.array(
                [
                    [0.2, 1.1, -3.4, 2.5],
                    [5.0, -1.5, 0.0, 4.2],
                    [-2.2, 3.3, 1.7, -0.8],
                ]
            ),
            np.array([2.0, -3.0, 1.0]),
            np.array([4.0, 2.0, -1.0, 3.0]),
            np.array([2, 0]),
            np.array([3, 1]),
        ),
    ]

    worst = 0.0
    for W, b, x, rows, cols in cases:
        try:
            got = sol.pruned_shell_forward(W, b, x, rows, cols)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(W, b, x, rows, cols)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
