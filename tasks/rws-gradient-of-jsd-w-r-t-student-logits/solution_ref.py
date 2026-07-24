import numpy as np


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def jsd_grad_wrt_student_logits(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    beta: float,
) -> np.ndarray:
    p = _softmax(teacher_logits)
    q = _softmax(student_logits)
    m = beta * p + (1.0 - beta) * q

    g = (1.0 - beta) * np.log(q / m)          # dJSD/dq
    return q * (g - np.sum(q * g))            # softmax-Jacobian-vector product
