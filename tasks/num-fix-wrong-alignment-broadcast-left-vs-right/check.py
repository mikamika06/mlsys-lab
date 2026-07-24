import numpy as np


def _oracle(a, b):
    return np.asarray(a) + np.asarray(b)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.arange(24, dtype=np.float64).reshape(2, 3, 4),
            np.arange(4, dtype=np.float64),
        ),
        (
            np.arange(6, dtype=np.float64).reshape(2, 3),
            np.array([10.0, 20.0, 30.0]),
        ),
        (
            np.ones((2, 1, 5), dtype=np.float64),
            np.arange(3, dtype=np.float64).reshape(3, 1),
        ),
        (
            np.arange(12, dtype=np.float64).reshape(3, 1, 4),
            np.arange(4, dtype=np.float64),
        ),
    ]

    ok = 1.0
    for a, b in cases:
        try:
            got = sol.broadcast_add_right(a, b)
            ref = _oracle(a, b)
        except Exception:
            ok = 0.0
            break
        if not np.array_equal(np.asarray(got), ref):
            ok = 0.0
            break
        if np.asarray(got).shape != ref.shape:
            ok = 0.0
            break

    return {"exact_match": ok}
