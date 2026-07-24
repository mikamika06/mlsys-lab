import numpy as np
from mlsys.scorers import max_abs_err

def _reference(logits, bias_map):
    out = logits.copy()
    for token, value in bias_map.items():
        if 0 <= token < out.shape[1]:
            out[:, token] += value
    return out

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_error = 0.0
    for _ in range(5):
        n = rng.integers(2, 10)
        d = rng.integers(3, 15)
        logits = rng.standard_normal((n, d))
        num_biases = rng.integers(0, d + 1)
        tokens = rng.choice(d, size=num_biases, replace=False)
        values = rng.normal(size=num_biases)
        bias_map = dict(zip(tokens, values))
        try:
            got = sol.apply_logit_bias_map(logits, bias_map)
        except Exception:
            return {"max_abs_err": 1.0}
        ref = _reference(logits, bias_map)
        err = max_abs_err(ref, got)
        if err > max_error:
            max_error = err
    return {"max_abs_err": max_error}
