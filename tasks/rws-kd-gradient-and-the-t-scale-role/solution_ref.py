import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def kd_gradient(student_logits, teacher_logits, labels, T, scale_t2=True):
    student_logits = np.asarray(student_logits, dtype=np.float64)
    teacher_logits = np.asarray(teacher_logits, dtype=np.float64)

    pt = _softmax(teacher_logits / T)
    ps = _softmax(student_logits / T)

    grad = (ps - pt) / T
    if scale_t2:
        grad = grad * (T * T)
    return grad
