import numpy as np


def _make_states(q, keys, values, splits):
    states = []
    for start, end in splits:
        k = keys[start:end]
        v = values[start:end]
        logits = q @ k.T
        m = np.max(logits, axis=1)
        weights = np.exp(logits - m[:, None])
        l = np.sum(weights, axis=1)
        o = weights @ v
        states.append((m.astype(np.float64), l.astype(np.float64), o.astype(np.float64)))
    return states


def _oracle(q, keys, values):
    logits = q @ keys.T
    m = np.max(logits, axis=1)
    p = np.exp(logits - m[:, None])
    return (p @ values) / np.sum(p, axis=1)[:, None]


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0.2, -0.4], [1.1, 0.3]], dtype=np.float64),
            np.array([[0.5, 0.2], [-0.3, 0.8], [1.0, -0.7]], dtype=np.float64),
            np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], dtype=np.float64),
            [(0, 1), (1, 3)],
        ),
        (
            np.array([[0.1, 0.7, -0.2]], dtype=np.float64),
            np.array([[0.3, -0.5, 0.4], [0.8, 0.2, -0.1], [-0.4, 0.9, 0.6], [0.2, 0.1, -0.8]], dtype=np.float64),
            np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0], [4.0, 2.0, -2.0], [1.0, 1.5, 3.0]], dtype=np.float64),
            [(0, 2), (2, 4)],
        ),
        (
            np.arange(12, dtype=np.float64).reshape(3, 4) / 5.0,
            np.array([[0.2, 0.1, 0.0, 0.4], [0.7, -0.2, 0.3, -0.5], [0.6, 0.9, -0.1, 0.2], [-0.3, 0.4, 0.8, 0.1], [0.5, -0.7, 0.2, 0.6]], dtype=np.float64),
            np.arange(25, dtype=np.float64).reshape(5, 5) / 7.0,
            [(0, 1), (1, 3), (3, 5)],
        ),
    ]

    best = float("inf")
    for q, k, v, splits in cases:
        states = _make_states(q, k, v, splits)
        ref = _oracle(q, k, v)
        try:
            got = np.asarray(sol.reconstruct_output(states), dtype=np.float64)
            err = float(np.max(np.abs(got - ref)))
        except Exception:
            return {"max_abs_err": float("inf")}
        best = min(best, err)
        if err >= 1e-6:
            return {"max_abs_err": err}
    return {"max_abs_err": best}
