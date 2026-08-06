import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_error = 0.0
    for _ in range(5):
        vocab_size = rng.integers(10, 50)
        dim = rng.integers(4, 12)
        weights = rng.standard_normal((vocab_size, dim)).astype(np.float64)
        n = rng.integers(1, 20)
        ids = rng.integers(0, vocab_size, size=n)
        ref = np.take(weights, ids, axis=0).astype(np.float64)
        try:
            got = sol.lookup_embeddings(ids.tolist(), weights.tolist())
        except Exception:
            return {"max_abs_err": float("inf")}
        err = max_abs_err(ref, np.array(got))
        if err > max_error:
            max_error = err
    return {"max_abs_err": max_error}
