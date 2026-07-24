import numpy as np
from mlsys import scorers


def _ref_expand(kv, num_q_heads):
    groups = num_q_heads // kv.shape[1]
    return np.repeat(kv, groups, axis=1)


def grade(sol, fx) -> dict:
    cases = [
        (1, 2, 4, 3, 2),
        (2, 4, 8, 5, 1),
        (3, 3, 6, 2, 4),
        (1, 8, 16, 1, 3),
    ]
    errs = []
    rng = np.random.default_rng(12345)
    for b, hkv, hq, s, d in cases:
        kv = rng.normal(size=(b, hkv, s, d)).astype(np.float32)
        expected = _ref_expand(kv, hq)
        try:
            got = sol.expand_kv_heads(kv, hq)
            err = scorers.max_abs_err(expected, got)
        except Exception:
            err = float("inf")
        errs.append(err)
    return {"max_abs_err": float(max(errs))}
