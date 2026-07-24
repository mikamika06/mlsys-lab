import numpy as np

def wasserstein_1_loss_on_sorted_logits(teacher_logits: np.ndarray,
                                        student_logits: np.ndarray) -> float:
    t = teacher_logits.ravel()
    s = student_logits.ravel()
    if len(t) < len(s):
        t = np.concatenate([t, np.zeros(len(s)-len(t))])
    elif len(s) < len(t):
        s = np.concatenate([s, np.zeros(len(t)-len(s))])
    t_sorted = np.sort(t)
    s_sorted = np.sort(s)
    return float(np.sum(np.abs(t_sorted - s_sorted)))
