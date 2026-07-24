import numpy as np


def grade(sol, fx) -> dict:
    cases = []

    cases.append(
        (
            np.array([
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, -1.0],
            ]),
            np.array([1.0, 2.0, 2.5, 0.0]),
        )
    )

    cases.append(
        (
            np.array([
                [1.0, 2.0, 0.5],
                [0.0, 1.0, 3.0],
                [2.0, 1.0, 1.0],
                [3.0, 0.0, 2.0],
                [1.0, -1.0, 1.5],
            ]),
            np.array([2.0, 1.0, 4.0, 3.0, 0.5]),
        )
    )

    n = 8
    hilbert = np.array(
        [[1.0 / (i + j + 1) for j in range(n)] for i in range(n + 3)],
        dtype=np.float64,
    )
    cases.append((hilbert, np.sin(np.arange(n + 3, dtype=np.float64))))

    worst = 0.0
    for A, b in cases:
        try:
            candidate = np.asarray(sol.least_squares_qr(A, b), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}

        reference = np.linalg.lstsq(A, b, rcond=None)[0]
        err = np.linalg.norm(candidate - reference) / (
            np.linalg.norm(reference) + 1e-12
        )
        worst = max(worst, float(err))

    return {"rel_err": worst}
