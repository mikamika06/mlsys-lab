import math
import numpy as np


def shape_constraint_loss(logits, target_sparsity, lam):
    logits = np.asarray(logits, dtype=np.float64)
    n = logits.size

    mask = np.empty_like(logits, dtype=np.float64)
    flat_logits = logits.ravel()
    flat_mask = mask.ravel()

    sum_mask = 0.0
    sum_sq_diff = 0.0

    for i in range(n):
        val = 1.0 / (1.0 + math.exp(-flat_logits[i]))
        flat_mask[i] = val
        sum_mask += val
        sum_sq_diff += (val - 0.5) ** 2

    sparsity = 1.0 - (sum_mask / n)
    task = sum_sq_diff / n
    constraint = lam * (sparsity - target_sparsity) ** 2
    loss = float(task + constraint)

    grad = np.empty_like(logits, dtype=np.float64)
    flat_grad = grad.ravel()

    d_sparsity_term = -2.0 * lam * (sparsity - target_sparsity) / n
    d_task_coeff = 2.0 / n

    for i in range(n):
        m_val = flat_mask[i]
        d_task_dm = d_task_coeff * (m_val - 0.5)
        d_constraint_dm = d_sparsity_term
        flat_grad[i] = (d_task_dm + d_constraint_dm) * m_val * (1.0 - m_val)

    return loss, np.asarray(grad, dtype=np.float64)
