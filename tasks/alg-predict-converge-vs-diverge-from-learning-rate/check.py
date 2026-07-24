import numpy as np


def _oracle(lrs, L):
    rates = np.asarray(lrs, dtype=np.float64)
    return (rates >= (2.0 / float(L))).astype(int).tolist()


def grade(sol, fx) -> dict:
    cases = [
        ([0.01, 0.05, 0.1, 0.2], 10.0),
        ([0.0, 0.199999, 0.2, 0.200001, 0.5], 10.0),
        ([0.01, 0.02, 0.05, 0.1], 20.0),
        ([0.5, 1.0, 1.5, 2.0], 1.0),
    ]

    for lrs, L in cases:
        expected = _oracle(lrs, L)
        try:
            got = sol.classify_learning_rates(list(lrs), L)
        except Exception:
            return {"exact_match": 0.0}

        if got != expected:
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
