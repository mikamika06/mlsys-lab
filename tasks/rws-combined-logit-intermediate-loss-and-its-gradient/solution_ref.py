import math


def _softmax(x):
    out = []
    for row in x:
        max_val = row[0]
        for val in row:
            if val > max_val:
                max_val = val
        sum_exp = 0.0
        exp_row = []
        for val in row:
            v = math.exp(val - max_val)
            exp_row.append(v)
            sum_exp += v
        out.append([v / sum_exp for v in exp_row])
    return out


def combined_logit_intermediate_loss(
    teacher_logits: list[list[float]],
    student_logits: list[list[float]],
    teacher_hidden: list[list[float]],
    student_hidden: list[list[float]],
    beta: float,
) -> tuple[float, list[list[float]], list[list[float]]]:
    p = _softmax(teacher_logits)
    q = _softmax(student_logits)

    loss_kl = 0.0
    for r_p, r_q in zip(p, q):
        for pi, qi in zip(r_p, r_q):
            loss_kl += pi * (math.log(pi) - math.log(qi))

    total_elements = sum(len(row) for row in student_hidden)
    sum_sq_diff = 0.0
    for r_sh, r_th in zip(student_hidden, teacher_hidden):
        for sh_val, th_val in zip(r_sh, r_th):
            diff = sh_val - th_val
            sum_sq_diff += diff * diff
    loss_hidden = beta * (sum_sq_diff / total_elements)

    grad_logits = []
    for r_p, r_q in zip(p, q):
        grad_row = []
        for pi, qi in zip(r_p, r_q):
            grad_row.append(qi - pi)
        grad_logits.append(grad_row)

    scale = 2.0 * beta / total_elements
    grad_hidden = []
    for r_sh, r_th in zip(student_hidden, teacher_hidden):
        grad_row = []
        for sh_val, th_val in zip(r_sh, r_th):
            grad_row.append(scale * (sh_val - th_val))
        grad_hidden.append(grad_row)

    return float(loss_kl + loss_hidden), grad_logits, grad_hidden
