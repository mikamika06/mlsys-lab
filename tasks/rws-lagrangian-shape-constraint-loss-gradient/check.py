import numpy as np


def _loss_only(logits, target_sparsity, lam):
    logits = np.asarray(logits, dtype=np.float64)
    mask = 1.0 / (1.0 + np.exp(-logits))
    sparsity = 1.0 - np.mean(mask)
    task = np.mean((mask - 0.5) ** 2)
    return float(task + lam * (sparsity - target_sparsity) ** 2)


def _oracle(logits, target_sparsity, lam):
    loss = _loss_only(logits, target_sparsity, lam)
    h = 1e-6
    grad = np.zeros_like(logits, dtype=np.float64)
    for i in range(logits.size):
        plus = np.array(logits, dtype=np.float64)
        minus = np.array(logits, dtype=np.float64)
        plus[i] += h
        minus[i] -= h
        grad[i] = (_loss_only(plus, target_sparsity, lam) -
                   _loss_only(minus, target_sparsity, lam)) / (2 * h)
    return loss, grad


def grade(sol, fx) -> dict:
    cases = [
        (np.array([-2.0, 0.0, 2.0]), 0.5, 3.0),
        (np.array([-1.5, -0.2, 0.7, 2.1]), 0.25, 1.7),
        (np.array([0.3, -0.8, 1.2, -2.4, 0.0]), 0.8, 5.0),
    ]

    worst = 0.0
    for logits, target, lam in cases:
        ref_loss, ref_grad = _oracle(logits, target, lam)
        try:
            got_loss, got_grad = sol.shape_constraint_loss(
                logits.copy(), target, lam
            )
            got_grad = np.asarray(got_grad, dtype=np.float64)
            values = np.concatenate([
                np.array([float(got_loss)], dtype=np.float64),
                got_grad.ravel()
            ])
            refs = np.concatenate([
                np.array([ref_loss], dtype=np.float64),
                ref_grad.ravel()
            ])
            err = float(np.max(np.abs(values - refs)))
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
