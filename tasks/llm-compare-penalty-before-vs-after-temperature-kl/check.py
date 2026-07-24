import numpy as np

def _apply_penalty_before(logits, penalty_fn, temperature):
    penalized = penalty_fn(logits)
    return penalized / temperature

def _apply_penalty_after(logits, penalty_fn, temperature):
    scaled = logits / temperature
    return penalty_fn(scaled)

def _mean_kl(logits1, logits2):
    exp1 = np.exp(logits1 - np.max(logits1, axis=-1, keepdims=True))
    p1 = exp1 / np.sum(exp1, axis=-1, keepdims=True)
    exp2 = np.exp(logits2 - np.max(logits2, axis=-1, keepdims=True))
    p2 = exp2 / np.sum(exp2, axis=-1, keepdims=True)
    kl_rows = np.sum(p1 * (np.log(p1 + 1e-12) - np.log(p2 + 1e-12)), axis=1)
    return float(np.mean(kl_rows))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = []
    for seed in [1, 2, 3]:
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=(5, 10))
        def penalty_fn(x):
            return x - 0.5
        temp = rng.uniform(0.5, 2.0)
        cases.append((logits, penalty_fn, temp))
    max_rel_err = 0.0
    for logits, penalty_fn, temp in cases:
        try:
            got = sol.compare_penalty_temperature(logits, penalty_fn, temp)
            before_logits = _apply_penalty_before(logits, penalty_fn, temp)
            after_logits = _apply_penalty_after(logits, penalty_fn, temp)
            ref = _mean_kl(before_logits, after_logits)
        except Exception:
            return {"rel_err": float("inf")}
        rel_err = abs(got - ref) / (abs(ref) + 1e-12)
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"rel_err": max_rel_err}
