import numpy as np


def _layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * gamma + beta


def _oracle(x, w_attn, w_ff, gamma, beta):
    h1 = x + _layer_norm(x, gamma, beta) @ w_attn
    y = h1 + _layer_norm(h1, gamma, beta) @ w_ff
    return y


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64),
            np.array([[1.0, 0.2], [-0.3, 0.8]], dtype=np.float64),
            np.array([[0.5, -0.1], [0.7, 0.4]], dtype=np.float64),
            np.array([1.0, 0.8], dtype=np.float64),
            np.array([0.1, -0.2], dtype=np.float64),
        ),
        (
            np.array([[0.2, -1.0, 3.0], [2.5, 1.5, -0.5]], dtype=np.float64),
            np.array([[0.4, 0.1, -0.2], [0.0, 0.6, 0.3], [0.5, -0.4, 0.2]], dtype=np.float64),
            np.array([[-0.3, 0.2, 0.7], [0.6, -0.1, 0.2], [0.1, 0.8, -0.5]], dtype=np.float64),
            np.array([0.9, 1.1, 0.7], dtype=np.float64),
            np.array([0.0, 0.2, -0.1], dtype=np.float64),
        ),
    ]

    worst = 0.0
    for x, wa, wf, gamma, beta in cases:
        try:
            got = np.asarray(sol.transformer_block(x, wa, wf, gamma, beta), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(x, wa, wf, gamma, beta)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
