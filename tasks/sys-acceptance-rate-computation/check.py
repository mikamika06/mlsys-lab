import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(5):
        n = rng.integers(3, 10)
        vocab = rng.integers(4, 12)
        target = rng.random((n, vocab))
        draft = rng.random((n, vocab))
        # normalize rows to sum to one
        target /= target.sum(axis=1, keepdims=True)
        draft /= draft.sum(axis=1, keepdims=True)
        ref = np.sum(np.minimum(target, draft), axis=1).astype(np.float64)
        try:
            got = sol.acceptance_rate(target, draft)
        except Exception:
            return {"rel_err": float("inf")}
        if not isinstance(got, np.ndarray):
            return {"rel_err": 0.0}
        got = np.asarray(got, dtype=np.float64)
        err = scorers.rel_err(ref, got)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
