import numpy as np


def _oracle(X, adapters, segments):
    n = X.shape[0]
    m = adapters[0].shape[1]
    out = np.zeros((n, m), dtype=np.float64)
    for start, end, adapter_id in segments:
        out[start:end] = X[start:end] @ adapters[adapter_id]
    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]),
            [
                np.array([[1., 0.5], [0.25, 1.]], dtype=np.float64),
                np.array([[2., -1.], [1., 3.]], dtype=np.float64),
            ],
            [(0, 2, 0), (2, 4, 1)],
        ),
        (
            np.array([
                [0.5, 1.5, -2.0],
                [3.0, -1.0, 4.0],
                [2.5, 2.5, 1.0],
                [-3.0, 0.5, 2.0],
                [1.0, -4.0, 3.0],
            ]),
            [
                np.array([[1., 0.], [0., 1.], [0.5, 0.5]], dtype=np.float64),
                np.array([[-1., 2.], [3., 1.], [0., -2.]], dtype=np.float64),
                np.array([[0.25, 1.], [1.5, -0.5], [2., 0.]], dtype=np.float64),
            ],
            [(0, 1, 2), (1, 4, 0), (4, 5, 1)],
        ),
    ]

    worst = 0.0
    for X, adapters, segments in cases:
        try:
            got = np.asarray(sol.sgmv(X, adapters, segments), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(X, adapters, segments)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
