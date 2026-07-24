import numpy as np
from mlsys.scorers import mean_kl

def _reference(logits, temps):
    logits = np.asarray(logits, dtype=np.float64)
    temps = np.asarray(temps, dtype=np.float64)
    probs = []
    for t in temps:
        scaled = logits / t
        max_val = np.max(scaled)
        exp_shifted = np.exp(scaled - max_val)
        probs.append(exp_shifted / np.sum(exp_shifted))
    return np.vstack(probs)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    kl_values = []
    for _ in range(5):
        n = rng.integers(3, 15)
        logits = rng.standard_normal(n)
        m = rng.integers(2, 6)
        temps = rng.uniform(0.1, 3.0, size=m)
        try:
            cand = sol.softmax_temperature_sweep(logits, temps)
            ref = _reference(logits, temps)
            kl = mean_kl(ref, cand)
        except Exception:
            return {"mean_kl": float("inf")}
        if np.isnan(kl):
            return {"mean_kl": float("inf")}
        kl_values.append(kl)
    avg_kl = float(np.mean(kl_values))
    return {"mean_kl": avg_kl}
