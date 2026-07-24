import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _make_state(logits, values):
    m = float(np.max(logits))
    w = np.exp(logits - m)
    l = float(np.sum(w))
    o = np.sum(w[:, None] * values, axis=0)
    return m, l, o


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1.0, -2.0, 3.0, 0.5]),
            np.array([[1.0, 0.0], [0.5, 2.0], [-1.0, 3.0], [2.0, -1.0]])
        ),
        (
            np.array([-10.0, -8.0, -9.0, 4.0, 5.0]),
            np.array([[2.0, 1.0], [1.0, -1.0], [0.0, 3.0], [4.0, 2.0], [-2.0, 1.0]])
        ),
        (
            np.array([12.0, 11.5, 10.0, -3.0, -4.0, 7.0]),
            np.array([[0.2, 1.1, -2.0], [2.0, 0.5, 1.0], [1.0, -1.0, 3.0],
                      [-2.0, 4.0, 0.0], [3.0, 1.0, -1.0], [1.5, 2.5, 0.5]])
        ),
    ]

    worst = 0.0
    for logits, values in cases:
        split = len(logits) // 2
        s1 = _make_state(logits[:split], values[:split])
        s2 = _make_state(logits[split:], values[split:])

        try:
            merged = sol.merge_mlo(s1, s2)
            got = np.asarray(merged[2], dtype=np.float64) / float(merged[1])
        except Exception:
            return {"max_abs_err": float("inf")}

        oracle = np.sum(
            _softmax(logits)[:, None] * values,
            axis=0,
        )
        worst = max(worst, float(np.max(np.abs(got - oracle))))

    return {"max_abs_err": worst}
