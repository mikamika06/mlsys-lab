import numpy as np


def _oracle(attn, k):
    scores = np.mean(np.asarray(attn, dtype=np.float64), axis=0)
    order = np.argsort(-scores, kind="stable")
    return np.sort(order[:k].astype(np.int64))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([
                [0.1, 0.8, 0.2, 0.4],
                [0.3, 0.6, 0.9, 0.2],
            ], dtype=np.float64),
            2,
        ),
        (
            np.array([
                [0.9, 0.1, 0.2, 0.3, 0.4],
                [0.8, 0.2, 0.3, 0.4, 0.5],
                [0.7, 0.3, 0.4, 0.5, 0.6],
            ], dtype=np.float64),
            3,
        ),
        (
            np.array([
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
                [0.2, 0.2, 0.5],
                [0.4, 0.4, 0.5],
            ], dtype=np.float64),
            2,
        ),
        (
            np.arange(30, dtype=np.float64).reshape(5, 6) / 30.0,
            4,
        ),
    ]

    ok = 1.0
    for attn, k in cases:
        try:
            got = np.asarray(sol.select_snapkv_indices(attn, k), dtype=np.int64)
        except Exception:
            ok = 0.0
            break
        if not np.array_equal(got, _oracle(attn, k)):
            ok = 0.0
            break
    return {"exact_match": ok}
