def _ref(logits, targets):
    import numpy as np
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    max_logits = np.max(logits, axis=1, keepdims=True)
    stable = logits - max_logits
    exp_sum = np.sum(np.exp(stable), axis=1, keepdims=True)
    logsumexp = np.log(exp_sum) + max_logits.squeeze()
    batch_idx = np.arange(logits.shape[0])
    target_logit = logits[batch_idx, targets]
    ce = - (target_logit - logsumexp)
    return float(np.mean(ce))

def grade(sol, fx):
    import numpy as np
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(5):
        N = rng.integers(2, 10)
        C = rng.integers(3, 8)
        logits = rng.standard_normal((N, C)).astype(np.float64)
        targets = rng.integers(0, C, size=N).astype(np.int64)
        try:
            got = sol.fused_cross_entropy(logits.tolist(), targets.tolist())
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _ref(logits.tolist(), targets.tolist())
        err = abs(got - ref)
        if err > max_err:
            max_err = err
    # also check type and shape
    if not isinstance(got, float) or np.isscalar(got) is False:
        return {"max_abs_err": float("inf")}
    return {"max_abs_err": max_err}
