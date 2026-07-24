import numpy as np


def _layernorm_forward(x, eps):
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _finite_difference_vjp(x, grad_y, eps):
    h = 1e-6
    ref = np.zeros_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xp = x.copy()
            xm = x.copy()
            xp[i, j] += h
            xm[i, j] -= h
            lp = np.sum(_layernorm_forward(xp, eps) * grad_y)
            lm = np.sum(_layernorm_forward(xm, eps) * grad_y)
            ref[i, j] = (lp - lm) / (2 * h)
    return ref


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 2.0, 4.0], [-2.0, 0.5, 3.0]], dtype=np.float64),
            np.array([[0.2, -0.5, 1.0], [1.5, -0.7, 0.1]], dtype=np.float64),
        ),
        (
            np.array([[0.1, -0.4, 1.2, 2.0]], dtype=np.float64),
            np.array([[1.0, 0.3, -0.2, 0.7]], dtype=np.float64),
        ),
        (
            np.arange(12, dtype=np.float64).reshape(3, 4) / 3.0,
            np.ones((3, 4), dtype=np.float64),
        ),
    ]

    worst = 0.0
    for x, gy in cases:
        try:
            got = np.asarray(sol.layernorm_vjp(x, gy, 1e-5), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _finite_difference_vjp(x, gy, 1e-5)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
