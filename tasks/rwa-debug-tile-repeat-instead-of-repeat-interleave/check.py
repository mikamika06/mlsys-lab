import numpy as np


def _oracle(kv, n_query_heads):
    batch, kv_heads, seq, dim = kv.shape
    repeat = n_query_heads // kv_heads
    return np.repeat(kv.astype(np.float64), repeat, axis=1)


def grade(sol, fx) -> dict:
    cases = [
        (1, 2, 4, 3, 4),
        (2, 3, 5, 2, 6),
        (1, 4, 3, 5, 8),
    ]

    max_err = 0.0
    for seed, kv_heads, seq, dim, query_heads in cases:
        rng = np.random.default_rng(seed)
        kv = rng.normal(size=(2, kv_heads, seq, dim)).astype(np.float32)

        try:
            got = sol.expand_kv_heads(kv, query_heads)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(kv, query_heads)

        if np.asarray(got).shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
