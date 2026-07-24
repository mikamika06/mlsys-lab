import numpy as np


def _oracle(mask, weights, grads, grow_count):
    del weights
    out = np.asarray(mask, dtype=np.int64).copy()
    zero_idx = np.flatnonzero(out == 0)
    count = min(int(grow_count), int(zero_idx.size))
    if count:
        scores = np.abs(np.asarray(grads, dtype=np.float64)[zero_idx])
        order = np.argsort(-scores, kind="stable")[:count]
        out[zero_idx[order]] = 1
    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1, 0, 0, 1, 0]),
            np.array([0.8, 9.0, -2.0, 0.1, 5.0]),
            np.array([0.1, 0.3, 0.9, 0.2, 0.4]),
            2,
        ),
        (
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([10.0, 1.0, 8.0, -7.0, 2.0, 6.0]),
            np.array([0.2, 0.01, 0.8, 0.7, 0.9, 0.6]),
            3,
        ),
        (
            np.array([1, 1, 0, 0, 0, 1, 0]),
            np.array([-5, 4, 100, -1, 3, 2, 50]),
            np.array([9, 8, 0.5, 0.4, 0.3, 7, 0.6]),
            1,
        ),
        (
            np.array([0, 0, 0, 0]),
            np.array([4, 3, 2, 1]),
            np.array([-0.1, -0.8, 0.4, 0.5]),
            10,
        ),
    ]

    exact = 1.0
    for mask, weights, grads, grow_count in cases:
        try:
            got = sol.rigl_grow(
                mask.copy(),
                weights.copy(),
                grads.copy(),
                grow_count,
            )
            got = np.asarray(got, dtype=np.int64)
        except Exception:
            exact = 0.0
            break

        expected = _oracle(mask, weights, grads, grow_count)
        if not np.array_equal(got, expected):
            exact = 0.0
            break
        if int(got.sum()) != int(expected.sum()):
            exact = 0.0
            break

    return {"exact_match": exact}
