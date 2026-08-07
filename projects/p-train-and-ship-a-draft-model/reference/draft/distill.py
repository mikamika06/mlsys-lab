import numpy as np

def distill_step(teacher_logits, student_logits):
    t_probs = np.exp(teacher_logits - np.max(teacher_logits))
    t_probs /= np.sum(t_probs)
    s_log_probs = student_logits - np.max(student_logits)
    s_log_probs -= np.log(np.sum(np.exp(s_log_probs)))
    kl = np.sum(t_probs * (np.log(t_probs + 1e-10) - s_log_probs))
    return float(kl)
