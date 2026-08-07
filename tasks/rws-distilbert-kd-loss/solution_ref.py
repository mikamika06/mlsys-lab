import math

def kd_loss(
    teacher_logits: list[list[float]],
    student_logits: list[list[float]],
    labels: list[int],
    alpha: float = 0.5,
    temperature: float = 1.0
) -> float:
    eps = 1e-12
    N = len(teacher_logits)
    C = len(teacher_logits[0])
    kl_sum = 0.0
    ce_sum = 0.0

    for i in range(N):
        t_row = [val / temperature for val in teacher_logits[i]]
        t_m = t_row[0]
        for j in range(1, C):
            if t_row[j] > t_m:
                t_m = t_row[j]

        pt_row = [math.exp(t_row[j] - t_m) for j in range(C)]
        pt_sum = sum(pt_row)
        pt_row = [val / pt_sum for val in pt_row]

        s_row = [val / temperature for val in student_logits[i]]
        s_m = s_row[0]
        for j in range(1, C):
            if s_row[j] > s_m:
                s_m = s_row[j]

        ps_row = [math.exp(s_row[j] - s_m) for j in range(C)]
        ps_sum = sum(ps_row)
        ps_row = [val / ps_sum for val in ps_row]

        row_kl = 0.0
        for j in range(C):
            p = pt_row[j]
            row_kl += p * (math.log(p + eps) - math.log(ps_row[j] + eps))
        kl_sum += row_kl

        label_idx = int(labels[i])
        ce_sum += -math.log(ps_row[label_idx] + eps)

    kl = kl_sum / N
    ce = ce_sum / N

    loss = alpha * temperature**2 * kl + (1.0 - alpha) * ce
    return float(loss)
