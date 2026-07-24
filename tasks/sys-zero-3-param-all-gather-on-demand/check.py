import numpy as np


def _oracle(weight_shards, x, grad_y):
    full_w = np.concatenate([np.asarray(s, dtype=np.float64) for s in weight_shards], axis=0)
    grad_w = np.asarray(grad_y, dtype=np.float64).T @ np.asarray(x, dtype=np.float64)
    split_sizes = [s.shape[0] for s in weight_shards]
    result = []
    start = 0
    for size in split_sizes:
        result.append(grad_w[start:start + size].copy())
        start += size
    return result


def _max_abs_err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[5.0, 6.0]]),
            ],
            np.array([[1.0, -1.0], [0.5, 2.0]]),
            np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 1.0]]),
        ),
        (
            [
                np.arange(12, dtype=np.float64).reshape(3, 4),
                np.arange(12, 24, dtype=np.float64).reshape(3, 4),
            ],
            np.array([[2.0, 0.0, -1.0, 1.0]]),
            np.ones((1, 6)),
        ),
        (
            [
                np.array([[0.2, -0.5, 1.5]]),
                np.array([[2.0, 3.0, -1.0], [4.0, 0.0, 0.5]]),
                np.array([[-2.0, 1.0, 2.5]]),
            ],
            np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]]),
            np.array([[0.1, 0.2, 0.3, 0.4], [1.0, -1.0, 2.0, 0.5]]),
        ),
    ]

    best = float("inf")
    for shards, x, grad_y in cases:
        try:
            got = sol.zero3_linear_backward(
                [s.copy() for s in shards],
                x.copy(),
                grad_y.copy(),
            )
            got_full = np.concatenate([np.asarray(g) for g in got], axis=0)
            ref_full = np.concatenate(_oracle(shards, x, grad_y), axis=0)
            best = min(best, _max_abs_err(got_full, ref_full))
        except Exception:
            return {"max_abs_err": float("inf")}

    return {"max_abs_err": best}
