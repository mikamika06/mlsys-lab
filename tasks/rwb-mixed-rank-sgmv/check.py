import numpy as np


def _oracle(x, adapter_ids, adapters):
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        a, b = adapters[int(adapter_ids[i])]
        out[i] = x[i] + b @ (a @ x[i])
    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0], [4.0, 1.0, 0.0]], dtype=np.float64),
            np.array([0, 1, 2]),
            [
                (
                    np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
                    np.array([[0.5], [0.0], [1.0]], dtype=np.float64),
                ),
                (
                    np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]], dtype=np.float64),
                    np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float64),
                ),
                (
                    np.array([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64),
                    np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -0.5]], dtype=np.float64),
                ),
            ],
        ),
        (
            np.arange(24, dtype=np.float64).reshape(6, 4) / 7.0,
            np.array([2, 0, 2, 1, 0, 1]),
            [
                (
                    np.ones((1, 4), dtype=np.float64),
                    np.ones((4, 1), dtype=np.float64) * 0.1,
                ),
                (
                    np.eye(4, dtype=np.float64)[:2],
                    np.eye(4, dtype=np.float64)[:, :2],
                ),
                (
                    np.array(
                        [[1, 0, 2, 0], [0, -1, 0, 1], [0.5, 0, 0, 1]],
                        dtype=np.float64,
                    ),
                    np.array(
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -1, 0]],
                        dtype=np.float64,
                    ),
                ),
            ],
        ),
    ]

    score = 0.0
    for x, adapter_ids, adapters in cases:
        try:
            got = sol.mixed_rank_sgmv(x, adapter_ids, adapters)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(x, adapter_ids, adapters)
        score = max(score, float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref))))
    return {"max_abs_err": score}
