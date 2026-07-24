import numpy as np


def _oracle_rate(A):
    vals = np.linalg.eigvals(A)
    vals = np.sort(np.abs(vals))[::-1]
    return float(vals[1] / vals[0])


def grade(sol, fx) -> dict:
    cases = [
        np.array([[10.0, 0.0], [0.0, 8.0]]),
        np.array([[12.0, 0.0], [0.0, 9.0]]),
        np.array([[20.0, 0.0], [0.0, 18.0]]),
    ]

    errors = []
    for A in cases:
        try:
            got = float(sol.estimate_convergence_rate(A))
        except Exception:
            return {"rel_err": 1.0}

        ref = _oracle_rate(A)
        errors.append(abs(got - ref) / (abs(ref) + 1e-12))

    return {"rel_err": float(max(errors))}
