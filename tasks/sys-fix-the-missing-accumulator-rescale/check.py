import numpy as np
from mlsys.scorers import rel_err

def _ref_softmax(scores):
    e = np.exp(scores)
    return e / e.sum()

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(10):
        size = rng.integers(2, 20)
        scores = rng.uniform(-1000.0, 1000.0, size=size)
        try:
            probs, _ = sol.streaming_softmax(scores)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _ref_softmax(scores)
        err = rel_err(ref, probs)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
