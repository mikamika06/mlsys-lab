import numpy as np


def _ref(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n = logits.shape[0]
    m = np.max(logits, axis=1, keepdims=True)
    shifted = logits - m
    lse = m[:, 0] + np.log(np.sum(np.exp(shifted), axis=1))
    log_probs = logits - lse[:, None]
    idx = np.arange(n)
    loss = -float(np.mean(log_probs[idx, targets]))
    probs = np.exp(log_probs)
    dlogits = probs.copy()
    dlogits[idx, targets] -= 1.0
    dlogits /= n
    return loss, dlogits


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    max_loss_err = 0.0
    max_grad_err = 0.0
    for _ in range(6):
        n = int(rng.integers(2, 12))
        c = int(rng.integers(2, 9))
        scale = float(rng.uniform(1.0, 50.0))
        logits = (rng.standard_normal((n, c)) * scale).astype(np.float64)
        targets = rng.integers(0, c, size=n).astype(np.int64)
        ref_loss, ref_grad = _ref(logits, targets)
        try:
            got_loss, got_grad = sol.fused_log_softmax_nll(logits, targets)
            got_loss = float(got_loss)
            got_grad = np.asarray(got_grad, dtype=np.float64)
        except Exception:
            return {"max_abs_err_loss": float("inf"), "max_abs_err_grad": float("inf")}
        if got_grad.shape != ref_grad.shape:
            return {"max_abs_err_loss": float("inf"), "max_abs_err_grad": float("inf")}
        max_loss_err = max(max_loss_err, abs(got_loss - ref_loss))
        max_grad_err = max(max_grad_err, float(np.max(np.abs(got_grad - ref_grad))))
    return {"max_abs_err_loss": max_loss_err, "max_abs_err_grad": max_grad_err}
