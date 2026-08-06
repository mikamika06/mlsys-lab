import numpy as np


def _reference(X, Wq, Wk, Wv):
    q = np.matmul(X, Wq)
    k = np.matmul(X, Wk)
    v = np.matmul(X, Wv)
    return q, k, v


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    X_np = rng.normal(size=(8, 16)).astype(np.float64)
    Wq_np = rng.normal(size=(16, 12)).astype(np.float64)
    Wk_np = rng.normal(size=(16, 12)).astype(np.float64)
    Wv_np = rng.normal(size=(16, 12)).astype(np.float64)

    X = X_np.tolist()
    Wq = Wq_np.tolist()
    Wk = Wk_np.tolist()
    Wv = Wv_np.tolist()

    ref = _reference(X_np, Wq_np, Wk_np, Wv_np)

    calls = {"count": 0}

    # Intercept matmul call from solution if solution defines or imports one
    original_matmul = getattr(sol, "matmul", None)

    if original_matmul is not None:
        def counted_matmul(*args, **kwargs):
            calls["count"] += 1
            return original_matmul(*args, **kwargs)
        sol.matmul = counted_matmul
    else:
        # Fallback tracking if matmul isn't directly exposed on module root
        calls["count"] = 1

    try:
        got = sol.fused_qkv_projection(X, Wq, Wk, Wv)
    except Exception:
        got = None
    finally:
        if original_matmul is not None:
            sol.matmul = original_matmul

    err = float("inf")
    if isinstance(got, tuple) and len(got) == 3:
        try:
            got_np = tuple(np.array(g, dtype=np.float64) for g in got)
            err = float(max(np.max(np.abs(a - b)) for a, b in zip(got_np, ref)))
        except Exception:
            err = float("inf")

    return {
        "max_abs_err": err,
        "matmul_calls": float(calls["count"])
    }
