import numpy as np
from mlsys import scorers


def _oracle(X, W):
    w_absmax = np.max(np.abs(W), axis=1)
    w_absmax = np.where(w_absmax == 0.0, 1.0, w_absmax)
    scale_w = w_absmax / 127.0
    W_q = np.clip(np.round(W / scale_w[:, None]), -127, 127).astype(np.int32)

    x_min = float(np.min(X))
    x_max = float(np.max(X))
    if x_max == x_min:
        x_max = x_min + 1e-8
    scale_x = (x_max - x_min) / 255.0
    zero_point = np.clip(np.round(-x_min / scale_x), 0, 255)
    X_q = np.clip(np.round(X / scale_x + zero_point), 0, 255).astype(np.int32)

    acc = (X_q - zero_point) @ W_q.T
    return acc.astype(np.float64) * scale_x * scale_w[None, :]


def grade(sol, fx) -> dict:
    """
    Random (X, W) pairs; runs the dynamic int8 quantize/matmul/dequantize
    pipeline as a NumPy oracle and compares against the submission.
    """
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(6):
        b = int(rng.integers(2, 10))
        d_in = int(rng.integers(2, 16))
        d_out = int(rng.integers(2, 10))

        X = rng.standard_normal((b, d_in)) * rng.uniform(0.5, 5.0)
        W = rng.standard_normal((d_out, d_in)) * rng.uniform(0.5, 5.0)

        expected = _oracle(X, W)
        try:
            got = np.asarray(sol.int8_linear_forward(X.copy(), W.copy()), dtype=np.float64)
            if got.shape != expected.shape:
                return {"rel_err": float("inf")}
            err = scorers.rel_err(expected, got)
        except Exception:
            return {"rel_err": float("inf")}

        worst = max(worst, err)

    return {"rel_err": worst}
