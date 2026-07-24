import numpy as np


def _oracle(student_logits, on_policy_targets, off_policy_targets, lam):
    x = np.asarray(student_logits, dtype=np.float64)
    on = np.asarray(on_policy_targets, dtype=np.int64)
    off = np.asarray(off_policy_targets, dtype=np.int64)
    m = np.max(x, axis=1, keepdims=True)
    log_z = m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))
    on_ce = -x[np.arange(x.shape[0]), on] + log_z[:, 0]
    off_ce = -x[np.arange(x.shape[0]), off] + log_z[:, 0]
    return float(np.mean(lam * on_ce + (1.0 - lam) * off_ce))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            np.array([0, 1]),
            np.array([1, 0]),
            0.5,
        ),
        (
            np.array(
                [[1.0, -1.0, 0.5], [0.2, 0.3, -0.7], [4.0, 0.0, -2.0]],
                dtype=np.float32,
            ),
            np.array([2, 0, 1]),
            np.array([0, 1, 2]),
            0.8,
        ),
        (
            np.array([[10.0, -5.0, 0.0, 1.0], [-3.0, 2.0, 7.0, 0.5]]),
            np.array([0, 2]),
            np.array([3, 1]),
            0.25,
        ),
    ]
    worst = 0.0
    for logits, on, off, lam in cases:
        try:
            got = float(sol.gkd_mixed_loss(logits, on, off, lam))
        except Exception:
            return {"rel_err": 1.0}
        ref = _oracle(logits, on, off, lam)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)
    return {"rel_err": worst}
