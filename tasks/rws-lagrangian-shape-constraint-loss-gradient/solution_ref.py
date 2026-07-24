import numpy as np


def shape_constraint_loss(logits, target_sparsity, lam):
    logits = np.asarray(logits, dtype=np.float64)
    mask = 1.0 / (1.0 + np.exp(-logits))
    n = logits.size

    sparsity = 1.0 - np.mean(mask)
    task = np.mean((mask - 0.5) ** 2)
    constraint = lam * (sparsity - target_sparsity) ** 2
    loss = float(task + constraint)

    d_task_dm = 2.0 * (mask - 0.5) / n
    d_constraint_dm = (
        -2.0 * lam * (sparsity - target_sparsity) / n
    )
    grad = (d_task_dm + d_constraint_dm) * mask * (1.0 - mask)

    return loss, np.asarray(grad, dtype=np.float64)
