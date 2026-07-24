import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def kd_hard_label_grad(student_logits, teacher_probs, label, temperature):
    p_kd = _softmax(student_logits / temperature)
    kd_grad = temperature * (p_kd - np.asarray(teacher_probs, dtype=np.float64))

    p_ce = _softmax(student_logits)
    y = np.zeros_like(p_ce)
    y[int(label)] = 1.0
    ce_grad = p_ce - y

    return kd_grad.astype(np.float64), ce_grad.astype(np.float64)
