import numpy as np


def _left_fold_reference(values):
    total = np.float64(0.0)
    for value in values:
        total = np.float64(total + np.float64(value))
    return float(total)


def _relative_error(a, b):
    return abs(float(a) - float(b)) / (abs(float(b)) + 1e-12)


def _cases():
    cases = [
        np.array([1e16, -1e16] + [1e-3] * 10000, dtype=np.float64),
        np.concatenate(
            [
                np.array([1e16], dtype=np.float64),
                np.ones(100000, dtype=np.float64),
                np.array([-1e16], dtype=np.float64),
            ]
        ),
    ]

    rng = np.random.default_rng(123)
    magnitudes = np.power(
        10.0, rng.integers(-12, 12, size=20000)
    )
    signs = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float64),
        size=20000,
    )
    cases.append((magnitudes * signs).astype(np.float64))

    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for values in _cases():
        reference = _left_fold_reference(values)
        try:
            got = sol.sequential_sum(values)
        except Exception:
            return {"rel_err": float("inf")}
        worst = max(worst, _relative_error(got, reference))
    return {"rel_err": worst}
