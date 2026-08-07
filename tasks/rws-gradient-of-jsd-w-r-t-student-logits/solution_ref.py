import math


def _softmax(z):
    n = len(z)

    max_val = z[0]
    for i in range(1, n):
        if z[i] > max_val:
            max_val = z[i]

    e = [0.0] * n
    sum_e = 0.0
    for i in range(n):
        val = math.exp(z[i] - max_val)
        e[i] = val
        sum_e += val

    return [val / sum_e for val in e]


def jsd_grad_wrt_student_logits(
    teacher_logits: list[float],
    student_logits: list[float],
    beta: float,
) -> list[float]:
    p = _softmax(teacher_logits)
    q = _softmax(student_logits)
    n = len(p)

    m = [0.0] * n
    for i in range(n):
        m[i] = beta * p[i] + (1.0 - beta) * q[i]

    g = [0.0] * n
    for i in range(n):
        g[i] = (1.0 - beta) * math.log(q[i] / m[i])

    sum_qg = 0.0
    for i in range(n):
        sum_qg += q[i] * g[i]

    result = [0.0] * n
    for i in range(n):
        result[i] = q[i] * (g[i] - sum_qg)

    return result
