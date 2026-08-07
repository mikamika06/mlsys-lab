import numpy as np


def hard_loss(logits, targets):
    exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    N = logits.shape[0]
    core = -np.log(np.clip(probs[np.arange(N), targets], 1e-15, 1.0))
    return float(np.mean(core))


def soft_loss(student_logits, teacher_logits, temperature):
    s_scaled = student_logits / temperature
    t_scaled = teacher_logits / temperature
    s_exps = np.exp(s_scaled - np.max(s_scaled, axis=-1, keepdims=True))
    s_probs = s_exps / np.sum(s_exps, axis=-1, keepdims=True)
    t_exps = np.exp(t_scaled - np.max(t_scaled, axis=-1, keepdims=True))
    t_probs = t_exps / np.sum(t_exps, axis=-1, keepdims=True)
    kl = np.sum(t_probs * (np.log(np.clip(t_probs, 1e-15, 1.0)) - np.log(np.clip(s_probs, 1e-15, 1.0))), axis=-1)
    return float(np.mean(kl) * (temperature ** 2))


def blended_loss(logits, targets, student_logits, teacher_logits, alpha, temperature):
    h = hard_loss(logits, targets)
    s = soft_loss(student_logits, teacher_logits, temperature)
    return float(alpha * h + (1.0 - alpha) * s)
