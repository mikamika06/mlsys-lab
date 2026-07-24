import numpy as np


def _oracle_softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float64),
        np.array([
            [-1200.0, -1199.0, -1198.0, -1197.0],
            [800.0, 0.0, -800.0, 400.0],
        ], dtype=np.float64),
        np.array([
            [50.0, 51.0, 52.0, 53.0, 54.0],
            [-900.0, -850.0, -800.0, -750.0, -700.0],
        ], dtype=np.float64),
    ]

    worst = 0.0
    for logits in cases:
        ref = _oracle_softmax(logits)
        try:
            got = sol.stable_softmax(logits.copy())
        except Exception:
            return {"rel_err": float("inf")}
        got = np.asarray(got, dtype=np.float64)
        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"rel_err": float("inf")}
        worst = max(worst, _rel_err(got, ref))
    return {"rel_err": worst}
