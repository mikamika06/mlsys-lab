import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def kd_loss_and_grad(teacher_logits, student_logits, T):
    teacher_logits = np.asarray(teacher_logits, dtype=np.float64)
    student_logits = np.asarray(student_logits, dtype=np.float64)

    batch = teacher_logits.shape[0]
    p = _softmax(teacher_logits / T)
    q = _softmax(student_logits / T)

    loss = T * T * (-np.sum(p * np.log(q)) / batch)
    grad = T * (q - p) / batch

    return float(loss), grad
