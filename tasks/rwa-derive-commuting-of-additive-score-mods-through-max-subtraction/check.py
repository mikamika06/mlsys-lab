import numpy as np


def _oracle(scores, alibi, window, soft_cap):
    x = np.asarray(scores, dtype=np.float64) + np.asarray(alibi, dtype=np.float64)
    x = soft_cap * np.tanh(x / soft_cap)

    n = x.shape[0]
    mask = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > window
    x = x.copy()
    x[mask] = -np.inf

    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    e[~np.isfinite(x)] = 0.0
    return e / np.sum(e, axis=1, keepdims=True)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 2.0], [0.5, 0.0]], dtype=np.float64),
            np.array([[0.0, -0.2], [0.1, 0.0]], dtype=np.float64),
            1,
            2.0,
        ),
        (
            np.array(
                [[3.0, -1.0, 0.5], [0.0, 2.0, -2.0], [1.5, 0.2, 4.0]],
                dtype=np.float64,
            ),
            np.array(
                [[0.0, -0.5, -1.0], [0.2, 0.0, -0.4], [0.1, -0.3, 0.0]],
                dtype=np.float64,
            ),
            1,
            1.5,
        ),
        (
            np.arange(16, dtype=np.float64).reshape(4, 4) / 3.0,
            np.flipud(np.arange(16, dtype=np.float64).reshape(4, 4)) / 10.0,
            0,
            3.0,
        ),
    ]

    worst = 0.0
    for scores, alibi, window, soft_cap in cases:
        try:
            got = sol.fused_attention_scores(scores, alibi, window, soft_cap)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(scores, alibi, window, soft_cap)
        worst = max(worst, float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref))))
    return {"max_abs_err": worst}
