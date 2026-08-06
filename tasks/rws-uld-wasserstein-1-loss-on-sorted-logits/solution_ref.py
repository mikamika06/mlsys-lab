import numpy as np

def wasserstein_1_loss_on_sorted_logits(teacher_logits: np.ndarray,
                                        student_logits: np.ndarray) -> float:
    t = teacher_logits.ravel()
    s = student_logits.ravel()
    t_list = list(t)
    s_list = list(s)
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
