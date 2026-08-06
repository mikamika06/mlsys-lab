import math
import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    batch = x.shape[0]
    classes = x.shape[1]
    out = np.zeros((batch, classes), dtype=np.float64)
    for i in range(batch):
        max_val = x[i, 0]
        for j in range(1, classes):
            if x[i, j] > max_val:
                max_val = x[i, j]
        row_sum = 0.0
        for j in range(classes):
            val = math.exp(x[i, j] - max_val)
            out[i, j] = val
            row_sum += val
        for j in range(classes):
            out[i, j] /= row_sum
    return out


def kd_loss_and_grad(teacher_logits, student_logits, T):
    teacher_logits = np.asarray(teacher_logits, dtype=np.float64)
    student_logits = np.asarray(student_logits, dtype=np.float64)

    batch = teacher_logits.shape[0]
    classes = teacher_logits.shape[1]

    teacher_scaled = np.zeros((batch, classes), dtype=np.float64)
    for i in range(batch):
        for j in range(classes):
            teacher_scaled[i, j] = teacher_logits[i, j] / T

    student_scaled = np.zeros((batch, classes), dtype=np.float64)
    for i in range(batch):
        for j in range(classes):
            student_scaled[i, j] = student_logits[i, j] / T

    p = _softmax(teacher_scaled)
    q = _softmax(student_scaled)

    sum_p_log_q = 0.0
    for i in range(batch):
        for j in range(classes):
            sum_p_log_q += p[i, j] * math.log(q[i, j])

    loss = T * T * (-sum_p_log_q / batch)
    grad = np.zeros((batch, classes), dtype=np.float64)
    for i in range(batch):
        for j in range(classes):
            grad[i, j] = T * (q[i, j] - p[i, j]) / batch

    return float(loss), grad
