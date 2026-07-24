import numpy as np


def _loss(logits, values, target):
    logits = np.asarray(logits, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(shifted)
    p /= np.sum(p, axis=1, keepdims=True)
    mask = p @ values
    return float(np.sum((mask - target) ** 2))


def _finite_difference_grad(logits, values, target, eps=1e-6):
    logits = np.asarray(logits, dtype=np.float64)
    out = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            plus = logits.copy()
            minus = logits.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            out[i, j] = (_loss(plus, values, target) - _loss(minus, values, target)) / (2 * eps)
    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[2.0, 0.0, -1.0], [0.5, -0.2, 1.1]], dtype=np.float64),
            np.array([1.0, 0.0, 0.5], dtype=np.float64),
            np.array([0.7, 0.2], dtype=np.float64),
        ),
        (
            np.array([[4.0, -3.0, 0.1, 2.0]], dtype=np.float64),
            np.array([0.2, 0.8, -0.5, 1.3], dtype=np.float64),
            np.array([0.4], dtype=np.float64),
        ),
        (
            np.array([[-1.0, 0.0, 1.0], [3.0, 2.0, -2.0]], dtype=np.float64),
            np.array([0.6, -0.4, 0.9], dtype=np.float64),
            np.array([0.1, 0.8], dtype=np.float64),
        ),
    ]

    worst = 0.0
    for logits, values, target in cases:
        try:
            got = np.asarray(sol.expected_mask_grad(logits, values, target), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _finite_difference_grad(logits, values, target)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
