import numpy as np
from mlsys.scorers import rel_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    max_error = 0.0
    for _ in range(5):
        n = rng.integers(2, 2000)
        concurrency = rng.integers(1, 10000, size=n).astype(np.float64)
        latency = rng.uniform(1e-3, 10.0, size=n).astype(np.float64)
        try:
            got = sol.compute_throughput(concurrency, latency)
        except Exception:
            return {"rel_err": float("inf")}
        ref = concurrency / latency
        err = rel_err(ref, got)
        if err > max_error:
            max_error = err
    return {"rel_err": max_error}
