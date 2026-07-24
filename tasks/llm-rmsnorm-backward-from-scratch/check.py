import numpy as np


def _forward(x, eps):
    r = np.sqrt(np.mean(x * x) + eps)
    return x / r


def _finite_difference_grad(x, grad_y, eps):
    out = np.zeros_like(x, dtype=np.float64)
    h = 1e-6
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        lp = float(np.sum(_forward(xp, eps) * grad_y))
        lm = float(np.sum(_forward(xm, eps) * grad_y))
        out[i] = (lp - lm) / (2.0 * h)
    return out


def grade(sol, fx) -> dict:
    cases = [
        (np.array([1.0, 2.0, 3.0]), np.array([0.5, -1.0, 2.0]), 1e-5),
        (np.array([-2.0, 0.5, 4.0, 1.0]), np.array([1.0, 0.0, -0.5, 2.0]), 1e-5),
        (np.array([0.1, -0.3]), np.array([3.0, -2.0]), 1e-3),
        (np.array([5.0, -1.0, 0.0, 2.0, -4.0]), np.array([-1.0, 2.0, 1.0, 0.5, -0.2]), 1e-5),
    ]

    worst = 0.0
    for x, grad_y, eps in cases:
        ref = _finite_difference_grad(x, grad_y, eps)
        try:
            got = np.asarray(sol.rmsnorm_backward(x, grad_y, eps), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}
        if got.shape != ref.shape:
            return {"rel_err": float("inf")}
        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        worst = max(worst, float(err))
    return {"rel_err": worst}
