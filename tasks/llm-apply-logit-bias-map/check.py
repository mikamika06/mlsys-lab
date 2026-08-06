import numpy as np
from mlsys.scorers import max_abs_err

def _reference(logits, bias_map):
    out = [row[:] for row in logits]
    d = len(out[0]) if out else 0
    for token, value in bias_map.items():
        if 0 <= token < d:
            for i in range(len(out)):
                out[i][token] += value
    return out

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_error = 0.0
    for _ in range(5):
        n = int(rng.integers(2, 10))
        d = int(rng.integers(3, 15))
        logits_np = rng.standard_normal((n, d))
        logits = logits_np.tolist()
        num_biases = int(rng.integers(0, d + 1))
        tokens = rng.choice(d, size=num_biases, replace=False)
        values = rng.normal(size=num_biases)
        bias_map = {int(t): float(v) for t, v in zip(tokens, values)}
        try:
            got = sol.apply_logit_bias_map(logits, bias_map)
        except Exception:
            return {"max_abs_err": 1.0}
        ref = _reference(logits, bias_map)
        err = max_abs_err(np.array(ref), np.array(got))
        if err > max_error:
            max_error = err
    return {"max_abs_err": max_error}
