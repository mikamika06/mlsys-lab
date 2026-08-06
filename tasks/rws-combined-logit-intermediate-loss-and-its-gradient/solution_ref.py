import math
import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    out = np.zeros(shape, dtype=np.float64)
    if len(shape) == 0:
        out[...] = 1.0
        return out
    prefix_shape = shape[:-1]
    last_dim = shape[-1]
    for idx in np.ndindex(prefix_shape) if prefix_shape else [()]:
        row = x[idx + (slice(None),)]
        max_val = row[0]
        for val in row:
            if val > max_val:
                max_val = val
        exp_row = np.zeros(last_dim, dtype=np.float64)
        sum_exp = 0.0
        for i in range(last_dim):
            val = math.exp(row[i] - max_val)
            exp_row[i] = val
            sum_exp += val
        for i in range(last_dim):
            out[idx + (i,)] = exp_row[i] / sum_exp
    return out


def combined_logit_intermediate_loss(
    teacher_logits,
    student_logits,
    teacher_hidden,
    student_hidden,
    beta,
):
    tl = np.asarray(teacher_logits, dtype=np.float64)
    sl = np.asarray(student_logits, dtype=np.float64)
    th = np.asarray(teacher_hidden, dtype=np.float64)
    sh = np.asarray(student_hidden, dtype=np.float64)

    p = _softmax(tl)
    q = _softmax(sl)

    loss_kl = 0.0
    for idx in np.ndindex(p.shape):
        pi = p[idx]
        qi = q[idx]
        loss_kl += pi * (math.log(pi) - math.log(qi))

    sum_sq_diff = 0.0
    for idx in np.ndindex(sh.shape):
        diff = sh[idx] - th[idx]
        sum_sq_diff += diff * diff
    loss_hidden = beta * (sum_sq_diff / sh.size)

    grad_logits = np.zeros(q.shape, dtype=np.float64)
    for idx in np.ndindex(q.shape):
        grad_logits[idx] = q[idx] - p[idx]

    grad_hidden = np.zeros(sh.shape, dtype=np.float64)
    scale = 2.0 * beta / sh.size
    for idx in np.ndindex(sh.shape):
        grad_hidden[idx] = scale * (sh[idx] - th[idx])

    return float(loss_kl + loss_hidden), grad_logits, grad_hidden
