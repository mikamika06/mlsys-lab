import numpy as np


def _oracle(x, W, b):
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.max(np.abs(x)) / 127.0
    if scale == 0:
        x_hat = np.zeros_like(x)
    else:
        q = np.clip(np.round(x / scale), -127, 127)
        x_hat = q * scale
    return x_hat @ W.T + b


def grade(sol, fx) -> dict:
    x_first = np.array([
        [0.12, -0.08, 0.04],
        [0.10, 0.05, -0.02],
    ], dtype=np.float64)

    x_shifted = np.array([
        [8.5, -6.0, 7.0],
        [10.0, 4.0, -9.0],
        [-7.5, 5.5, 6.5],
    ], dtype=np.float64)

    W = np.array([
        [0.8, -0.4, 0.3],
        [-0.2, 0.9, 0.5],
    ], dtype=np.float64)
    b = np.array([0.1, -0.05], dtype=np.float64)

    try:
        sol.quantized_linear_dynamic(x_first, W, b)
        got = sol.quantized_linear_dynamic(x_shifted, W, b)
    except Exception:
        return {"rel_err": float("inf")}

    ref = _oracle(x_shifted, W, b)
    got = np.asarray(got, dtype=np.float64)

    if got.shape != ref.shape:
        return {"rel_err": float("inf")}

    err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
    return {"rel_err": float(err)}
