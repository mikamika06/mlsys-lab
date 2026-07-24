import numpy as np


def _make_spd(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    A = M @ M.T + np.eye(n) * 0.5
    b = rng.normal(size=n)
    return A, b


def _ref(A, b):
    return np.linalg.solve(A, b)


def grade(sol, fx) -> dict:
    cases = [
        _make_spd(2, 1),
        _make_spd(5, 2),
        _make_spd(12, 3),
        _make_spd(20, 4),
    ]

    worst = 0.0
    for A, b in cases:
        try:
            got = np.asarray(sol.solve_spd(A, b), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}

        ref = _ref(A, b)
        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        worst = max(worst, float(err))

    return {"rel_err": worst}
