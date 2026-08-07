import numpy as np


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    s_soft = np.exp(student_logits / temperature) / np.sum(np.exp(student_logits / temperature), axis=-1, keepdims=True)
    t_soft = np.exp(teacher_logits / temperature) / np.sum(np.exp(teacher_logits / temperature), axis=-1, keepdims=True)

    kd = np.sum(-t_soft * np.log(s_soft + 1e-8)) * (temperature ** 2)
    ce = np.mean((student_logits - labels) ** 2)

    return float(alpha * kd + (1 - alpha) * ce)
