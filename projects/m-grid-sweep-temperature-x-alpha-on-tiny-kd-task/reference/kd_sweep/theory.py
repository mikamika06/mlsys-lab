import numpy as np
from kd_sweep.analysis import softmax


def verify_gradient_factor(student_logits, teacher_logits, T, alpha):
    p_s = softmax(student_logits / T)
    p_t = softmax(teacher_logits / T)

    eps = 1e-5
    grad_num = np.zeros_like(student_logits)
    for i in range(student_logits.shape[0]):
        for j in range(student_logits.shape[1]):
            student_logits[i, j] += eps
            p_s_hi = softmax(student_logits / T)
            loss_hi = -np.sum(p_t * np.log(np.clip(p_s_hi, 1e-12, 1.0))) * (T ** 2)
            student_logits[i, j] -= 2 * eps
            p_s_lo = softmax(student_logits / T)
            loss_lo = -np.sum(p_t * np.log(np.clip(p_s_lo, 1e-12, 1.0))) * (T ** 2)
            student_logits[i, j] += eps
            grad_num[i, j] = (loss_hi - loss_lo) / (2 * eps)

    p_s_full = softmax(student_logits / T)
    grad_anal = (p_s_full - p_t) * T

    rel_err = np.linalg.norm(grad_num - grad_anal) / (np.linalg.norm(grad_anal) + 1e-8)
    return float(rel_err)


def compute_effective_temperature_shift(teacher_logits, T, noise_std):
    p_t = softmax(teacher_logits / T)
    noisy_logits = teacher_logits + noise_std * np.ones_like(teacher_logits)
    p_t_noisy = softmax(noisy_logits / T)
    shift = np.linalg.norm(p_t_noisy - p_t)
    return float(shift)
