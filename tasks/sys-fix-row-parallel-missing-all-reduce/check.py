import numpy as np


def _oracle(x_shards, w_shards, bias):
    x = np.concatenate(x_shards, axis=1)
    w = np.concatenate(w_shards, axis=0)
    return x.astype(np.float64) @ w.astype(np.float64) + bias.astype(np.float64)


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[5.0, 6.0], [7.0, 8.0]])
            ],
            [
                np.array([[1.0, 0.5], [0.0, 1.0]]),
                np.array([[0.5, 1.0], [1.5, -0.5]])
            ],
            np.array([0.25, -0.75]),
        ),
        (
            [
                np.array([[2.0, -1.0, 3.0]]),
                np.array([[4.0]]),
                np.array([[5.0, 6.0]])
            ],
            [
                np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0]]),
                np.array([[3.0, -2.0]]),
                np.array([[1.0, 1.0], [-1.0, 0.5]])
            ],
            np.array([1.0, 2.0]),
        ),
    ]

    worst = 0.0
    for x_shards, w_shards, bias in cases:
        ref = _oracle(x_shards, w_shards, bias)
        try:
            got = np.asarray(
                sol.row_parallel_linear(x_shards, w_shards, bias),
                dtype=np.float64,
            )
        except Exception:
            return {"rel_err": float("inf")}
        err = np.linalg.norm(got.ravel() - ref.ravel()) / (
            np.linalg.norm(ref.ravel()) + 1e-12
        )
        worst = max(worst, float(err))
    return {"rel_err": worst}
