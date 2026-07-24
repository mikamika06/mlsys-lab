import numpy as np

def _ref(logits):
    max_vals = logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(logits - max_vals)
    probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    return probs

def grade(sol, fx) -> dict:
    eps = 1e-12
    rng = np.random.default_rng(42)
    cases = []
    # Large logits that can overflow in a naive implementation
    for _ in range(3):
        logits = rng.normal(size=(5, 8)) * 1000 + 500
        cases.append(logits)
    # Moderate logits
    for _ in range(2):
        logits = rng.normal(size=(4, 6))
        cases.append(logits)

    kl_total = 0.0
    for logits in cases:
        try:
            cand = sol.stable_softmax(np.array(logits, dtype=np.float64))
        except Exception:
            return {"mean_kl": float("inf")}
        ref = _ref(np.array(logits, dtype=np.float64))

        # Compute KL divergence per row
        kl_rows = np.sum(ref * (np.log(ref + eps) - np.log(cand + eps)), axis=1)
        kl_total += kl_rows.mean()

    avg_kl = kl_total / len(cases)
    return {"mean_kl": float(avg_kl)}
