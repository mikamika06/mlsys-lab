import numpy as np
from mlsys import scorers


def _oracle(kv, num_query_heads):
    repeat = num_query_heads // kv.shape[1]
    return np.repeat(kv, repeat, axis=1)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(1234)
    cases = [
        (2, 2, 3, 4, 8),
        (1, 3, 5, 2, 12),
        (2, 4, 2, 3, 16),
    ]

    errors = []
    for b, hkv, t, d, hq in cases:
        kv = rng.normal(size=(b, hkv, t, d)).astype(np.float32)
        ref = _oracle(kv, hq)
        try:
            got = sol.expand_kv_heads(kv, hq)
            errors.append(scorers.rel_err(ref, got))
        except Exception:
            return {"rel_err": float("inf")}

    return {"rel_err": float(max(errors))}
