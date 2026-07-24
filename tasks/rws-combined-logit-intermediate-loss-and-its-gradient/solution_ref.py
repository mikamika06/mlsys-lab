import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


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

    loss_kl = np.sum(p * (np.log(p) - np.log(q)))
    loss_hidden = beta * np.mean((sh - th) ** 2)

    grad_logits = (q - p)
    grad_hidden = (2.0 * beta / sh.size) * (sh - th)

    return float(loss_kl + loss_hidden), grad_logits, grad_hidden
