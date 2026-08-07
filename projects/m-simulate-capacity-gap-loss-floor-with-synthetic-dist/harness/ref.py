import numpy as np


def generate_teacher_logits(num_samples=100, num_classes=10, seed=42):
    rng = np.random.RandomState(seed)
    u = rng.randn(num_samples, num_classes)
    s = np.linspace(10.0, 1.0, min(num_samples, num_classes))
    vt = rng.randn(num_classes, num_classes)
    return np.dot(u * s, vt)


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


def detect_mode_collapse(student_logits_history, entropy_threshold):
    collapsed_steps = []
    for t, logits in enumerate(student_logits_history):
        max_l = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        probs = np.clip(probs, 1e-12, 1.0)
        entropies = -np.sum(probs * np.log(probs), axis=-1)
        mean_entropy = np.mean(entropies)
        if mean_entropy < entropy_threshold:
            collapsed_steps.append(t)
    return collapsed_steps


def derive_effective_temperature(teacher_logits, target_temperature, confidence_alpha):
    max_l = np.max(teacher_logits, axis=-1, keepdims=True)
    exp_l = np.exp(teacher_logits - max_l)
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

    max_p = np.max(probs, axis=-1)
    avg_top1 = np.mean(max_p)

    effective_temp = target_temperature * (1.0 + confidence_alpha * (avg_top1 - 0.5))
    return float(np.maximum(effective_temp, 0.1))
