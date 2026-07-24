import numpy as np
from mlsys.scorers import max_abs_err

def _stable_ce(logits, targets):
    m = np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(logits - m)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    log_probs = logits - m - np.log(sum_exp)
    ce = -log_probs[np.arange(len(targets)), targets]
    return float(np.mean(ce))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = []

    # Small batches
    for N, C in [(5, 3), (10, 4)]:
        logits = rng.normal(size=(N, C))
        targets = rng.integers(low=0, high=C, size=N)
        cases.append((logits, targets))

    # Medium batches with moderate values
    for N, C in [(50, 10), (100, 20)]:
        logits = rng.uniform(-5, 5, size=(N, C))
        targets = rng.integers(0, C, size=N)
        cases.append((logits, targets))

    # Large values to test numerical stability
    for N, C in [(3, 4), (2, 6)]:
        logits = rng.normal(loc=1e4, scale=1e3, size=(N, C))
        targets = rng.integers(0, C, size=N)
        cases.append((logits, targets))

    max_err = 0.0
    for logits, targets in cases:
        try:
            got = sol.cross_entropy_from_logits(logits, targets)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _stable_ce(logits, targets)
        err = max_abs_err(ref, got)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
