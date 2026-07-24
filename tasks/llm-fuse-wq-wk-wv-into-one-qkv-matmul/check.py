import numpy as np


def _reference(X, Wq, Wk, Wv):
    q = np.matmul(X, Wq)
    k = np.matmul(X, Wk)
    v = np.matmul(X, Wv)
    return q, k, v


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    X = rng.normal(size=(8, 16)).astype(np.float64)
    Wq = rng.normal(size=(16, 12)).astype(np.float64)
    Wk = rng.normal(size=(16, 12)).astype(np.float64)
    Wv = rng.normal(size=(16, 12)).astype(np.float64)

    ref = _reference(X, Wq, Wk, Wv)

    calls = {"count": 0}
    original_matmul = np.matmul

    def counted_matmul(*args, **kwargs):
        calls["count"] += 1
        return original_matmul(*args, **kwargs)

    np.matmul = counted_matmul
    try:
        got = sol.fused_qkv_projection(X, Wq, Wk, Wv)
    except Exception:
        got = None
    finally:
        np.matmul = original_matmul

    err = float("inf")
    if isinstance(got, tuple) and len(got) == 3:
        try:
            err = float(max(np.max(np.abs(a - b)) for a, b in zip(got, ref)))
        except Exception:
            err = float("inf")

    return {
        "max_abs_err": err,
        "matmul_calls": float(calls["count"])
    }
