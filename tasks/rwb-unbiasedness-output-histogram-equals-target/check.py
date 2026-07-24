import numpy as np


def _oracle_target(p):
    return np.asarray(p, dtype=np.float64)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0.50, 0.30, 0.20]),
            np.array([0.40, 0.40, 0.20]),
            11,
            20000,
        ),
        (
            np.array([0.05, 0.10, 0.25, 0.60]),
            np.array([0.20, 0.20, 0.30, 0.30]),
            42,
            30000,
        ),
        (
            np.array([0.70, 0.10, 0.10, 0.10]),
            np.array([0.25, 0.25, 0.25, 0.25]),
            99,
            30000,
        ),
    ]

    worst = 0.0
    for p, q, seed, n in cases:
        try:
            got = sol.speculative_histogram(
                p.copy(), q.copy(), seed, n
            )
        except Exception:
            return {"rel_err": 1.0}

        got = np.asarray(got, dtype=np.float64)
        if got.shape != p.shape:
            return {"rel_err": 1.0}
        err = _rel_err(got, _oracle_target(p))
        worst = max(worst, err)

    return {"rel_err": worst}
