import numpy as np


def _quantize_int8(a):
    a = np.asarray(a, dtype=np.float32)
    mx = float(np.max(np.abs(a)))
    scale = mx / 127.0 if mx != 0 else 1.0
    q = np.round(a / scale).clip(-127, 127).astype(np.int8)
    return q, scale


def _oracle(X, W, outlier_cols):
    n = W.shape[1]
    mask = np.ones(n, dtype=bool)
    mask[outlier_cols] = False

    X_regular = X
    W_regular = W[:, mask]
    W_outlier = W[:, outlier_cols]

    x8, sx = _quantize_int8(X_regular)
    w8, sw = _quantize_int8(W_regular)

    regular = (x8.astype(np.int32) @ w8.astype(np.int32)).astype(np.float32)
    regular *= sx * sw

    outlier = (
        X.astype(np.float16) @ W_outlier.astype(np.float16)
    ).astype(np.float32)

    result = np.zeros((X.shape[0], n), dtype=np.float32)
    result[:, mask] = regular
    result[:, outlier_cols] = outlier
    return result


def _int8_baseline(X, W):
    x8, sx = _quantize_int8(X)
    w8, sw = _quantize_int8(W)
    return (x8.astype(np.int32) @ w8.astype(np.int32)).astype(np.float32) * sx * sw


def _rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = []

    for m, k, n in [(8, 16, 12), (5, 32, 20), (10, 24, 15)]:
        X = rng.normal(0, 1, size=(m, k)).astype(np.float32)
        W = rng.normal(0, 0.5, size=(k, n)).astype(np.float32)
        W[:, 2] *= 80
        W[:, 7] *= 60
        W[:, -1] *= 50
        cols = np.array([2, 7, n - 1], dtype=np.int64)
        cases.append((X, W, cols))

    max_err = 0.0
    all_better = 1.0

    for X, W, cols in cases:
        try:
            got = np.asarray(sol.mixed_precision_matmul(X, W, cols), dtype=np.float32)
        except Exception:
            return {"rel_err": 1.0, "beats_int8": 0.0}

        ref = _oracle(X, W, cols)
        baseline = _int8_baseline(X, W)

        max_err = max(max_err, _rel_err(got, ref))

        ref32 = X.astype(np.float32) @ W.astype(np.float32)
        if _rel_err(got, ref32) >= _rel_err(baseline, ref32):
            all_better = 0.0

    return {
        "rel_err": float(max_err),
        "beats_int8": float(all_better),
    }
