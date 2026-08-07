import math


def _softmax(x):
    result = []
    for row in x:
        max_val = max(row)
        exp_row = [math.exp(val - max_val) for val in row]
        sum_exp = sum(exp_row)
        result.append([e / sum_exp for e in exp_row])
    return result


def kd_loss_and_grad(teacher_logits, student_logits, T):
    batch = len(teacher_logits)
    C = len(teacher_logits[0])

    scaled_teacher = [[val / T for val in row] for row in teacher_logits]
    scaled_student = [[val / T for val in row] for row in student_logits]

    p = _softmax(scaled_teacher)
    q = _softmax(scaled_student)

    loss_sum = 0.0
    for i in range(batch):
        for j in range(C):
            loss_sum += p[i][j] * math.log(q[i][j])

    loss = T * T * (-loss_sum / batch)

    grad = []
    for i in range(batch):
        grad_row = []
        for j in range(C):
            val = (T / batch) * (q[i][j] - p[i][j])
            grad_row.append(val)
        grad.append(grad_row)

    return float(loss), grad
