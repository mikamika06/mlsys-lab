import numpy as np


def softmax(logits, temperature=1.0):
    """Compute temperature-scaled softmax probabilities."""
    scaled = logits / float(temperature)
    scaled -= np.max(scaled, axis=-1, keepdims=True)
    exp_l = np.exp(scaled)
    return exp_l / np.sum(exp_l, axis=-1, keepdims=True)


def compute_divergence(teacher_logits, student_logits, divergence_type="forward_kl", temperature=1.0):
    """Compute divergence loss between teacher and student logits."""
    p = softmax(teacher_logits, temperature)
    q = softmax(student_logits, temperature)
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)

    if divergence_type == "forward_kl":
        div = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=-1)
    elif divergence_type == "reverse_kl":
        div = np.sum(q_safe * (np.log(q_safe) - np.log(p_safe)), axis=-1)
    elif divergence_type == "jsd":
        m = 0.5 * (p_safe + q_safe)
        kl_pm = np.sum(p_safe * (np.log(p_safe) - np.log(m)), axis=-1)
        kl_qm = np.sum(q_safe * (np.log(q_safe) - np.log(m)), axis=-1)
        div = 0.5 * kl_pm + 0.5 * kl_qm
    else:
        raise ValueError(f"Unsupported divergence_type: {divergence_type}")

    return float(np.mean(div) * (float(temperature) ** 2))


def compute_gkd_step_loss(teacher_logits, student_logits, config):
    """Compute GKD step loss using GKDConfig parameters."""
    return compute_divergence(
        teacher_logits,
        student_logits,
        divergence_type=config.divergence_type,
        temperature=config.temperature,
    )
