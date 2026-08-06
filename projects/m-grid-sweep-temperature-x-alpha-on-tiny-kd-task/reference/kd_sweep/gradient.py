import numpy as np

def verify_gradient_scaling(teacher_logits, student_logits, T):
    eps = 1e-5
    s = student_logits.copy()
    s[0, 0] += eps

    p_t = np.exp((teacher_logits - np.max(teacher_logits, axis=-1, keepdims=True)) / T)
    p_t /= np.sum(p_t, axis=-1, keepdims=True)

    p_s_plus = np.exp((s - np.max(s, axis=-1, keepdims=True)) / T)
    p_s_plus /= np.sum(p_s_plus, axis=-1, keepdims=True)
    kl_plus = np.sum(p_t * (np.log(np.clip(p_t, 1e-12, 1.0)) - np.log(np.clip(p_s_plus, 1e-12, 1.0))), axis=-1).mean()
    loss_plus = (T ** 2) * kl_plus

    s[0, 0] -= 2 * eps
    p_s_minus = np.exp((s - np.max(s, axis=-1, keepdims=True)) / T)
    p_s_minus /= np.sum(p_s_minus, axis=-1, keepdims=True)
    kl_minus = np.sum(p_t * (np.log(np.clip(p_t, 1e-12, 1.0)) - np.log(np.clip(p_s_minus, 1e-12, 1.0))), axis=-1).mean()
    loss_minus = (T ** 2) * kl_minus

    num_grad = (loss_plus - loss_minus) / (2 * eps)
    return float(np.abs(num_grad))
