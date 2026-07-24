import numpy as np
from mlsys.scorers import mean_kl

def _ref(logits, mask):
    # Additive -inf masking
    mask_bool = mask.astype(bool)
    masked_logits = np.where(mask_bool, logits, -np.inf)
    max_per_row = np.max(masked_logits, axis=-1, keepdims=True)
    exp_vals = np.exp(masked_logits - max_per_row)
    sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)
    probs = exp_vals / sum_exp
    return probs

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(5):
        batch = rng.integers(1, 6)
        seq_len = rng.integers(3, 10)
        logits = rng.standard_normal((batch, seq_len))
        mask = rng.choice([True, False], size=(batch, seq_len))
        # Ensure at least one True per row
        for i in range(batch):
            if not mask[i].any():
                mask[i, rng.integers(0, seq_len)] = True
        cases.append((logits, mask))

    kl_total = 0.0
    for logits, mask in cases:
        try:
            cand = sol.masked_softmax(logits, mask)
        except Exception:
            return {"mean_kl": float("inf")}
        ref = _ref(logits, mask)
        kl_total += mean_kl(ref, cand)

    avg_kl = kl_total / len(cases)
    return {"mean_kl": avg_kl}
