import numpy as np


def _loss(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return float(np.mean(-np.log(probs[np.arange(len(labels)), labels])))


def _finite_difference(logits, labels):
    eps = 1e-6
    grad = np.zeros_like(logits, dtype=np.float64)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            plus = logits.copy()
            minus = logits.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            grad[i, j] = (_loss(plus, labels) - _loss(minus, labels)) / (2 * eps)
    return grad


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[2.0, 1.0, 0.0], [0.5, -1.0, 3.0]], dtype=np.float64),
            np.array([0, 2], dtype=np.int64),
        ),
        (
            np.array([[10.0, 9.0, 8.0, 7.0]], dtype=np.float64),
            np.array([3], dtype=np.int64),
        ),
        (
            np.array([[-2.0, 0.0, 1.5], [3.0, 3.0, -4.0]], dtype=np.float64),
            np.array([1, 0], dtype=np.int64),
        ),
    ]
    worst = 0.0
    for logits, labels in cases:
        try:
            got = sol.ce_backward(logits.copy(), labels.copy())
        except Exception:
            return {"rel_err": float("inf")}
        ref = _finite_difference(logits, labels)
        worst = max(worst, _rel_err(got, ref))
    return {"rel_err": worst}
