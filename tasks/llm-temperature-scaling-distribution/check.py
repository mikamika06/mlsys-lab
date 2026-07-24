import numpy as np
from mlsys import scorers

def _reference(logits, T):
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / T
    max_scaled = np.max(scaled, axis=-1, keepdims=True)
    exp_scaled = np.exp(scaled - max_scaled)
    probs = exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)
    return probs

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((5,))
    T = 0.75
    try:
        cand = sol.temperature_scale(logits, T)
    except Exception:
        return {"mean_kl": float("inf")}
    ref = _reference(logits, T)
    kl = scorers.mean_kl(ref, cand)
    return {"mean_kl": kl}
