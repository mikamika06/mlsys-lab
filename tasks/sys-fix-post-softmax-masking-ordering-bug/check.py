import numpy as np
from mlsys.scorers import rel_err

def _ref(logits, mask):
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, bool)
    masked = np.where(mask, -np.inf, logits)
    max_logits = np.max(masked, axis=-1, keepdims=True)
    exp_shift = np.exp(masked - max_logits)
    sum_exp = np.sum(exp_shift, axis=-1, keepdims=True)
    probs = exp_shift / sum_exp
    return probs

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    worst_err = 0.0
    for _ in range(10):
        batch = rng.integers(2,5)
        seq_len = rng.integers(3,8)
        logits = rng.standard_normal((batch, seq_len))
        mask = rng.choice([False, True], size=(batch, seq_len), p=[0.7, 0.3])
        try:
            got = sol.masked_softmax(logits, mask)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _ref(logits, mask)
        err = rel_err(ref, got)
        if err > worst_err:
            worst_err = err
    return {"rel_err": worst_err}
