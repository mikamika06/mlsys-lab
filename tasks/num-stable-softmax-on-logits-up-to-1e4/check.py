import numpy as np
from mlsys.scorers import mean_kl

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    kl_sum = 0.0
    n_cases = 5
    for _ in range(n_cases):
        n = rng.integers(2, 8)
        d = rng.integers(3, 10)
        logits = rng.uniform(-1e4, 1e4, size=(n, d))
        # Reference: numerically stable softmax
        max_vals = np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(logits - max_vals)
        ref = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        try:
            out = sol.stable_softmax(logits)
        except Exception:
            return {"mean_kl": float("inf")}
        if out.shape != logits.shape or out.dtype != np.float64:
            return {"mean_kl": float("inf")}
        kl_sum += mean_kl(ref, out)
    avg_kl = kl_sum / n_cases
    return {"mean_kl": avg_kl}
