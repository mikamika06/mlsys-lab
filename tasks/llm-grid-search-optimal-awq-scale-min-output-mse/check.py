import numpy as np


def _quantize(a):
    a = np.asarray(a, dtype=np.float64)
    z = np.max(np.abs(a)) / 127.0
    if z == 0:
        return np.zeros_like(a)
    return np.clip(np.round(a / z), -127, 127) * z


def _mse_for_scale(W, X, s):
    scaled = W * s[np.newaxis, :]
    q = _quantize(scaled)
    return np.mean((W @ X - (q / s[np.newaxis, :]) @ X) ** 2)


def _oracle(W, X, steps):
    col = np.max(np.abs(W), axis=0) + 1e-8
    best = None
    best_mse = float("inf")
    for alpha in np.linspace(0.0, 1.0, steps):
        s = col ** alpha
        mse = _mse_for_scale(W, X, s)
        if mse < best_mse:
            best_mse = mse
            best = s
    return best, best_mse


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0.2, 1.5, -3.0], [2.1, -0.7, 0.4]], dtype=np.float64),
            np.array([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.2]], dtype=np.float64),
            31,
        ),
        (
            np.array([[5.0, 0.1], [-2.0, 3.0], [1.0, -4.0]], dtype=np.float64),
            np.array([[0.3], [2.0]], dtype=np.float64),
            41,
        ),
        (
            np.array([[1.0, 4.0, 2.0], [-1.0, 0.5, 3.0]], dtype=np.float64),
            np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 1.5], [2.0, -1.0, 0.2]], dtype=np.float64),
            21,
        ),
    ]

    worst = 0.0
    for W, X, steps in cases:
        try:
            got_s = np.asarray(sol.awq_grid_scale(W, X, steps), dtype=np.float64)
        except Exception:
            return {"mse": float("inf")}

        _, ref_mse = _oracle(W, X, steps)
        got_mse = _mse_for_scale(W, X, got_s)
        worst = max(worst, float(max(0.0, got_mse - ref_mse)))

    return {"mse": worst}
