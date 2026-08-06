import numpy as np
from mlsys.scorers import mean_kl

def _ref(logits, mask):
    # Additive -inf masking
    mask_bool = np.array(mask).astype(bool)
    logits_arr = np.array(logits)
    masked_logits = np.where(mask_bool, logits_arr, -np.inf)
    max_per_row = np.max(masked_logits, axis=-1, keepdims=True)
    exp_vals = np.exp(masked_logits - max_per_row)
    sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)
    probs = exp_vals / sum_exp
    return probs

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(5):
        batch = int(rng.integers(1, 6))
        seq_len = int(rng.integers(3, 10))
        logits = rng.standard_normal((batch, seq_len)).tolist()
        mask_bool = rng.choice([True, False], size=(batch, seq_len))
        # Ensure at least one True per row
        for i in range(batch):
            if not mask_bool[i].any():
                mask_bool[i, rng.integers(0, seq_len)] = True
        mask = mask_bool.astype(int).tolist()
        cases.append((logits, mask))

    kl_total = 0.0
    for logits, mask in cases:
        try:
            cand_list = sol.masked_softmax(logits, mask)
            cand = np.array(cand_list)
        except Exception:
            return {"mean_kl": float("inf")}
        ref = _ref(logits, mask)
        kl_total += mean_kl(ref, cand)

    avg_kl = kl_total / len(cases)
    return {"mean_kl": avg_kl}
