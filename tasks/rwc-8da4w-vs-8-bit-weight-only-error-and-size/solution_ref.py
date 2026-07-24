import numpy as np


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def compare_linear_quantization(W: np.ndarray, X: np.ndarray) -> dict:
    W = np.asarray(W, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)

    y = X @ W.T

    sw8 = np.maximum(np.max(np.abs(W), axis=1, keepdims=True) / 127.0, 1e-12)
    q8 = np.round(W / sw8).clip(-127, 127).astype(np.int8)
    y8 = X @ (q8.astype(np.float32) * sw8).T

    sx = max(float(np.max(np.abs(X)) / 127.0), 1e-12)
    xq = np.round(X / sx).clip(-127, 127).astype(np.int8)

    sw4 = np.maximum(np.max(np.abs(W), axis=1, keepdims=True) / 7.0, 1e-12)
    q4 = np.round(W / sw4).clip(-7, 7).astype(np.int8)
    y4 = (xq.astype(np.float32) * sx) @ (q4.astype(np.float32) * sw4).T

    n, d = W.shape
    size8 = W.nbytes / float(n * d + 4 * n)
    size4 = W.nbytes / float((n * d + 1) // 2 + 4 * n)

    e4 = _rel_err(y, y4)
    e8 = _rel_err(y, y8)

    return {
        "error_8da4w": e4,
        "error_wo8": e8,
        "size_8da4w": size4,
        "size_wo8": size8,
        "tradeoff": 1.0 if size4 > size8 and e4 <= e8 else 0.0,
    }
