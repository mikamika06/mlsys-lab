import numpy as np


def softmax(x, axis=-1):
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex / np.sum(ex, axis=axis, keepdims=True)


def compute_kd_loss(student_logits, teacher_logits, labels, T, alpha):
    p_s = softmax(student_logits / T)
    p_t = softmax(teacher_logits / T)
    kd_loss = -np.sum(p_t * np.log(np.clip(p_s, 1e-12, 1.0))) * (T ** 2)

    p_s_hard = softmax(student_logits)
    ce_loss = -np.sum(np.log(np.clip(p_s_hard[np.arange(len(labels)), labels], 1e-12, 1.0)))

    total_loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
    return float(total_loss)
