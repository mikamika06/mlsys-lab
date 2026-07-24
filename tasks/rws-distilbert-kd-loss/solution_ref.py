import numpy as np

def kd_loss(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.5,
    temperature: float = 1.0
) -> float:
    eps = 1e-12

    # Temperature‑scaled softmax for teacher
    t_max = np.max(teacher_logits / temperature, axis=1, keepdims=True)
    pt = np.exp((teacher_logits / temperature) - t_max)
    pt /= np.sum(pt, axis=1, keepdims=True)

    # Temperature‑scaled softmax for student
    s_max = np.max(student_logits / temperature, axis=1, keepdims=True)
    ps = np.exp((student_logits / temperature) - s_max)
    ps /= np.sum(ps, axis=1, keepdims=True)

    # KL divergence (mean over batch)
    kl = np.mean(np.sum(pt * (np.log(pt + eps) - np.log(ps + eps)), axis=1))

    # Cross‑entropy with true labels
    ce = -np.mean(np.log(ps[np.arange(labels.size), labels] + eps))

    loss = alpha * temperature**2 * kl + (1.0 - alpha) * ce
    return float(loss)
