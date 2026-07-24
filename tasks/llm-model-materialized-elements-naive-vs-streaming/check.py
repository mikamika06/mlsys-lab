import numpy as np
from mlsys.scorers import rel_err

def _reference(logits):
    """Stable softmax reference implementation."""
    m = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = np.sum(e, axis=1, keepdims=True)
    return e / s

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    tests = []
    # 1. random normal
    tests.append(rng.standard_normal((5,4)))
    # 2. all zeros
    tests.append(np.zeros((3,6)))
    # 3. large positives
    tests.append(np.full((2,3), 1000.0))
    # 4. mixed large and small
    arr = rng.uniform(-1000, 1000, size=(4,5))
    tests.append(arr)
    # 5. single row with extreme values
    tests.append(np.array([[700, -700, 0]]))

    max_err = 0.0
    for logits in tests:
        try:
            got = sol.softmax_streaming(logits.astype(np.float64))
            ref = _reference(logits)
            err = rel_err(ref, got)
            if np.isnan(err):
                err = float('inf')
            if err > max_err:
                max_err = err
        except Exception:
            return {"rel_err": float("inf")}
    return {"rel_err": max_err}
