import numpy as np
from mlsys import scorers

def _ref(trace):
    tokens_per_step = trace.sum(axis=1)
    agg = tokens_per_step.mean()
    single = (tokens_per_step > 0).mean()
    return np.array([agg, single], dtype=np.float64)

def grade(sol, fx) -> dict:
    # Generate a variety of random traces
    rng = np.random.default_rng(42)
    cases = [
        np.zeros((5, 3), dtype=int),
        np.ones((4, 2), dtype=int),
        rng.integers(0, 2, size=(10, 5)),
        rng.integers(0, 2, size=(20, 7)),
        rng.integers(0, 2, size=(15, 1))
    ]
    max_err = 0.0
    for trace in cases:
        try:
            got = sol.throughput(trace)
            arr = np.asarray(got, dtype=np.float64).reshape(2,)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _ref(trace)
        err = scorers.rel_err(ref, arr)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
