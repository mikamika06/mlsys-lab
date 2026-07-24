import numpy as np

from mlsys import probe

_CASES = [
    dict(seed=0, n=5000, block=17, scale=5.0),
    dict(seed=1, n=3000, block=64, scale=50.0),
    dict(seed=2, n=2000, block=7, scale=500.0),
]


def _ref_lse(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def grade(sol, fx) -> dict:
    max_rel = 0.0
    total_ops = 0

    for c in _CASES:
        rng = np.random.default_rng(c["seed"])
        x = (rng.standard_normal(c["n"]) * c["scale"]).astype(np.float64)
        ref = _ref_lse(x)

        try:
            ops = probe.count_line_events(sol.streaming_lse, x, c["block"])
            got = float(sol.streaming_lse(x, c["block"]))
        except Exception:
            return {"rel_err": float("inf"), "op_count": 0.0}

        total_ops += ops
        if not np.isfinite(got):
            return {"rel_err": float("inf"), "op_count": float(total_ops)}
        rel = abs(got - ref) / (abs(ref) + 1e-12)
        max_rel = max(max_rel, rel)

    return {"rel_err": max_rel, "op_count": float(total_ops)}
