def wasserstein_1_loss_on_sorted_logits(teacher_logits: list[float],
                                        student_logits: list[float]) -> float:
    t_list = list(teacher_logits)
    s_list = list(student_logits)
    if len(t_list) < len(s_list):
        t_list.extend([0.0] * (len(s_list) - len(t_list)))
    elif len(s_list) < len(t_list):
        s_list.extend([0.0] * (len(t_list) - len(s_list)))
    t_sorted = sorted(t_list)
    s_sorted = sorted(s_list)
    total = 0.0
    for tv, sv in zip(t_sorted, s_sorted):
        total += abs(tv - sv)
    return float(total)
