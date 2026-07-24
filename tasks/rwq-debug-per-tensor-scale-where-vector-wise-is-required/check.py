import numpy as np


def _quantize_rows(A):
    A = np.asarray(A, dtype=np.float64)
    scales = np.max(np.abs(A), axis=1) / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    q = np.rint(A / scales[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scales


def _quantize_cols(A):
    A = np.asarray(A, dtype=np.float64)
    scales = np.max(np.abs(A), axis=0) / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    q = np.rint(A / scales[None, :])
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scales


def _reference(X, W):
    xq, xs = _quantize_rows(X)
    wq, ws = _quantize_cols(W)
    acc = xq.astype(np.int32) @ wq.astype(np.int32)
    return acc.astype(np.float64) * xs[:, None] * ws[None, :]


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    cases = [
        (
            np.array([[1.0, 2.0], [10.0, 1.0]]),
            np.array([[1.0, 3.0], [2.0, 1.0]]),
        ),
        (
            rng.normal(size=(8, 16)).astype(np.float64)
            * np.array([1, 2, 5, 10, 20, 1, 7, 15])[:, None],
            rng.normal(size=(16, 6)).astype(np.float64)
            * np.array([1, 3, 10, 2, 5, 20])[None, :],
        ),
        (
            rng.normal(size=(5, 9)).astype(np.float64),
            rng.normal(size=(9, 7)).astype(np.float64),
        ),
    ]

    worst = 0.0
    for X, W in cases:
        try:
            got = sol.int8_matmul_per_channel(X, W)
        except Exception:
            return {"rel_err": float("inf")}
        worst = max(worst, _rel_err(got, _reference(X, W)))
    return {"rel_err": worst}
