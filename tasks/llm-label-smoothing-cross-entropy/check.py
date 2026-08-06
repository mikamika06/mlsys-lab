import numpy as np

def _reference_loss(logits, targets, eps):
    N, K = logits.shape
    # stable log‑softmax
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(logits - logits_max)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    log_softmax = logits - logits_max - np.log(sum_exp)

    # smoothed target distribution
    y_onehot = np.zeros_like(logits)
    y_onehot[np.arange(N), targets] = 1.0
    y_smooth = (1 - eps) * y_onehot + eps / K

    loss_per_sample = -np.sum(y_smooth * log_softmax, axis=1)
    return float(np.mean(loss_per_sample))

def grade(sol, fx):
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(5):
        N = rng.integers(1, 10)
        K = rng.integers(2, 20)
        logits = rng.standard_normal((N, K))
        targets = rng.integers(0, K, size=N)
        eps = rng.uniform(0.0, 0.5)

        try:
            got = sol.label_smoothed_cross_entropy(logits.tolist(), targets.tolist(), eps)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _reference_loss(logits, targets, eps)
        err = abs(got - ref)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
