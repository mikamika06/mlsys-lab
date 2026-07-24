import numpy as np

def _reference(logits, targets, ignore_index):
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    # stable log‑softmax
    max_logits = logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(logits - max_logits)
    sum_exp = exp_shifted.sum(axis=1, keepdims=True)
    log_probs = logits - max_logits - np.log(sum_exp)
    neg_log_likelihood = -log_probs[np.arange(len(targets)), targets]
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    return float(neg_log_likelihood[mask].mean())

def grade(sol, fx) -> dict:
    func = getattr(sol, "masked_cross_entropy", None)
    if func is None:
        return {"max_abs_err": 1.0}
    rng = np.random.default_rng(seed=42)
    max_diff = 0.0
    for _ in range(5):
        N = rng.integers(2, 10)
        C = rng.integers(3, 8)
        logits = rng.standard_normal((N, C))
        # generate targets with some ignored positions
        targets = rng.integers(-1, C, size=N)  # -1 will be used as ignore_index
        ignore_index = -1
        try:
            got = func(logits, targets, ignore_index=ignore_index)
            ref = _reference(logits, targets, ignore_index)
            diff = abs(float(got) - float(ref))
            if diff > max_diff:
                max_diff = diff
        except Exception:
            return {"max_abs_err": 1.0}
    return {"max_abs_err": max_diff}
