import numpy as np


def compute_kl_divergence(p_logits, q_logits, temperature):
    p_scaled = p_logits / temperature
    q_scaled = q_logits / temperature

    p_max = np.max(p_scaled, axis=-1, keepdims=True)
    q_max = np.max(q_scaled, axis=-1, keepdims=True)

    p_exp = np.exp(p_scaled - p_max)
    p_probs = p_exp / np.sum(p_exp, axis=-1, keepdims=True)

    p_log_probs = p_scaled - p_max - np.log(np.sum(p_exp, axis=-1, keepdims=True))
    q_log_probs = q_scaled - q_max - np.log(np.sum(np.exp(q_scaled - q_max), axis=-1, keepdims=True))

    kl = np.sum(p_probs * (p_log_probs - q_log_probs), axis=-1)
    return np.mean(kl) * (temperature ** 2)


def simulate_capacity_loss_floor(teacher_logits, rank_constraint, temperature_grid):
    u, s, vt = np.linalg.svd(teacher_logits, full_matrices=False)
    s_constrained = s.copy()
    if rank_constraint < len(s):
        s_constrained[rank_constraint:] = 0.0
    student_logits = np.dot(u * s_constrained, vt)

    floors = []
    for temp in temperature_grid:
        kl = compute_kl_divergence(teacher_logits, student_logits, temp)
        floors.append(float(kl))
    return np.array(floors)
