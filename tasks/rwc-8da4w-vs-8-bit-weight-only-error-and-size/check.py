import numpy as np


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def _oracle(W, X):
    W = np.asarray(W, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)

    y = X @ W.T

    sw8 = np.maximum(np.max(np.abs(W), axis=1, keepdims=True) / 127.0, 1e-12)
    wq8 = np.round(W / sw8).clip(-127, 127).astype(np.int8)
    w8 = wq8.astype(np.float32) * sw8
    y8 = X @ w8.T

    sx = max(float(np.max(np.abs(X)) / 127.0), 1e-12)
    xq = np.round(X / sx).clip(-127, 127).astype(np.int8)

    sw4 = np.maximum(np.max(np.abs(W), axis=1, keepdims=True) / 7.0, 1e-12)
    wq4 = np.round(W / sw4).clip(-7, 7).astype(np.int8)
    w4 = wq4.astype(np.float32) * sw4
    y4 = (xq.astype(np.float32) * sx) @ w4.T

    n, d = W.shape
    size8 = W.nbytes / float(n * d + 4 * n)
    size4 = W.nbytes / float((n * d + 1) // 2 + 4 * n)

    e4 = _rel_err(y, y4)
    e8 = _rel_err(y, y8)
    trade = 1.0 if size4 > size8 and e4 <= e8 else 0.0

    return {
        "error_8da4w": e4,
        "error_wo8": e8,
        "size_8da4w": size4,
        "size_wo8": size8,
        "tradeoff": trade,
    }


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (rng.normal(size=(8, 16)).astype(np.float32),
         rng.normal(size=(5, 16)).astype(np.float32)),
        (rng.normal(size=(3, 7)).astype(np.float32),
         rng.normal(size=(4, 7)).astype(np.float32)),
        (rng.normal(scale=3, size=(11, 9)).astype(np.float32),
         rng.normal(scale=0.5, size=(6, 9)).astype(np.float32)),
    ]

    scores = {
        "error_8da4w": float("inf"),
        "error_wo8": float("inf"),
        "size_8da4w": 0.0,
        "size_wo8": 0.0,
        "tradeoff": 0.0,
    }

    for W, X in cases:
        ref = _oracle(W, X)
        try:
            got = sol.compare_linear_quantization(W, X)
        except Exception:
            return scores

        scores["error_8da4w"] = min(
            scores["error_8da4w"],
            abs(float(got["error_8da4w"]) - ref["error_8da4w"])
        )
        scores["error_wo8"] = min(
            scores["error_wo8"],
            abs(float(got["error_wo8"]) - ref["error_wo8"])
        )

        if abs(float(got["size_8da4w"]) - ref["size_8da4w"]) < 1e-12:
            scores["size_8da4w"] = 1.0
        else:
            scores["size_8da4w"] = 0.0

        if abs(float(got["size_wo8"]) - ref["size_wo8"]) < 1e-12:
            scores["size_wo8"] = 1.0
        else:
            scores["size_wo8"] = 0.0

        if float(got["tradeoff"]) == ref["tradeoff"]:
            scores["tradeoff"] = 1.0
        else:
            scores["tradeoff"] = 0.0

    return scores
