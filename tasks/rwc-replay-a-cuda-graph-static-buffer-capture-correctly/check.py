import numpy as np


def _oracle(W):
    W = np.asarray(W, dtype=np.float64)
    static_in = None
    static_out = None

    def capture_replay(X):
        nonlocal static_in, static_out
        X = np.asarray(X, dtype=np.float64)
        if static_in is None:
            static_in = np.empty_like(X)
            static_out = np.empty((X.shape[0], W.shape[0]), dtype=np.float64)
        np.copyto(static_in, X)
        np.copyto(static_out, static_in @ W.T)
        return static_out.copy()

    return capture_replay


def grade(sol, fx) -> dict:
    W = np.array(
        [
            [0.5, -1.0, 2.0],
            [1.5, 0.25, -0.75],
        ],
        dtype=np.float64,
    )
    xs = [
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        np.array([[4.0, -1.0, 0.5]], dtype=np.float64),
        np.array([[-2.0, 3.0, 1.0]], dtype=np.float64),
    ]

    ref_replay = _oracle(W)
    ref = [ref_replay(x) for x in xs]

    try:
        replay = sol.static_buffer_replay(W)
        got = [replay(x) for x in xs]
        err = max(
            float(np.max(np.abs(a - b)))
            for a, b in zip(ref, got)
        )
    except Exception:
        err = float("inf")

    return {"max_abs_err": err}
