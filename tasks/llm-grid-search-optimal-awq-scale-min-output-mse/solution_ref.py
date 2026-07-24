import numpy as np


def _quantize(a):
    a = np.asarray(a, dtype=np.float64)
    z = np.max(np.abs(a)) / 127.0
    if z == 0:
        return np.zeros_like(a)
    return np.clip(np.round(a / z), -127, 127) * z


def awq_grid_scale(W, X, steps=41):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    col = np.max(np.abs(W), axis=0) + 1e-8
    best = None
    best_mse = float("inf")

    for alpha in np.linspace(0.0, 1.0, steps):
        s = col ** alpha
        q = _quantize(W * s[np.newaxis, :])
        out = (q / s[np.newaxis, :]) @ X
        mse = np.mean((W @ X - out) ** 2)
        if mse < best_mse:
            best_mse = mse
            best = s

    return np.asarray(best, dtype=np.float64)
