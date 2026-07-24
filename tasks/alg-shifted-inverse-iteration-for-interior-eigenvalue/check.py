import numpy as np


def _oracle(A, sigma, iterations):
    values, vectors = np.linalg.eigh(A)
    idx = int(np.argmin(np.abs(values - sigma)))
    return float(values[idx])


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[2.0, 0.3, 0.0], [0.3, 5.0, 0.2], [0.0, 0.2, 9.0]]),
            4.8,
            25,
        ),
        (
            np.array([[8.0, -1.0, 0.0, 0.0], [-1.0, 3.0, 0.5, 0.0], [0.0, 0.5, 6.0, 0.4], [0.0, 0.0, 0.4, 11.0]]),
            5.7,
            30,
        ),
        (
            np.array([[1.0, 0.2], [0.2, 7.0]]),
            1.4,
            15,
        ),
    ]
    worst = 0.0
    for A, sigma, iterations in cases:
        ref = _oracle(A, sigma, iterations)
        try:
            got, vec = sol.shifted_inverse_iteration(A, sigma, iterations)
            got = float(got)
        except Exception:
            return {"rel_err": float("inf")}
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)
    return {"rel_err": float(worst)}
