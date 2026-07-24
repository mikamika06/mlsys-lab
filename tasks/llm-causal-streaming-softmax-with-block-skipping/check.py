import numpy as np
from mlsys.scorers import max_abs_err

def _ref(logits, mask):
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = logits.shape[0]
    out = np.zeros_like(logits, dtype=np.float64)
    for i in range(n):
        row = logits[i]
        m = mask[i]
        causal_mask = np.arange(n) <= i
        combined_mask = m & causal_mask
        if not combined_mask.any():
            continue
        vals = row[combined_mask]
        max_val = np.max(vals)
        exp_vals = np.exp(vals - max_val)
        probs = exp_vals / np.sum(exp_vals)
        out[i, combined_mask] = probs
    return out

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_error = 0.0
    for _ in range(5):
        n = rng.integers(3, 10)
        logits = rng.standard_normal((n, n))
        mask = rng.choice([True, False], size=(n, n), p=[0.8, 0.2])
        block_size = rng.integers(1, n + 1)
        try:
            got = sol.streaming_causal_softmax(logits, mask, block_size)
        except Exception:
            return {"max_abs_err": 1e9}
        ref = _ref(logits, mask)
        err = max_abs_err(ref, got)
        if err > max_error:
            max_error = err
    return {"max_abs_err": max_error}
