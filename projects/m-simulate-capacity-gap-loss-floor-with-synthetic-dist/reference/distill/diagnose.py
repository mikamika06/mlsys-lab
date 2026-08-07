import numpy as np


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
