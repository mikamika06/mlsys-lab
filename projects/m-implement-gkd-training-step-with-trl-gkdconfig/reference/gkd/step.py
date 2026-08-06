import numpy as np


class GKDConfig:
    def __init__(self, beta=0.0, lmbda=1.0, temperature=1.0):
        self.beta = beta
        self.lmbda = lmbda
        self.temperature = temperature


def compute_gkd_step(student_logits, teacher_logits, config):
    s_logits = np.array(student_logits, dtype=np.float64) / config.temperature
    t_logits = np.array(teacher_logits, dtype=np.float64) / config.temperature
    s_max = np.max(s_logits, axis=-1, keepdims=True)
    t_max = np.max(t_logits, axis=-1, keepdims=True)
    s_exp = np.exp(s_logits - s_max)
    t_exp = np.exp(t_logits - t_max)
    s_probs = s_exp / np.sum(s_exp, axis=-1, keepdims=True)
    t_probs = t_exp / np.sum(t_exp, axis=-1, keepdims=True)
    s_log_probs = s_logits - s_max - np.log(np.sum(s_exp, axis=-1, keepdims=True))
    t_log_probs = t_logits - t_max - np.log(np.sum(t_exp, axis=-1, keepdims=True))
    beta = config.beta
    if beta == 0.0:
        loss = np.sum(t_probs * (t_log_probs - s_log_probs), axis=-1)
    elif beta == 1.0:
        loss = np.sum(s_probs * (s_log_probs - t_log_probs), axis=-1)
    else:
        m_probs = (1.0 - beta) * t_probs + beta * s_probs
        m_log_probs = np.log(m_probs + 1e-12)
        term1 = np.sum(t_probs * (t_log_probs - m_log_probs), axis=-1)
        term2 = np.sum(s_probs * (s_log_probs - m_log_probs), axis=-1)
        loss = (1.0 - beta) * term1 + beta * term2
    grad = s_probs - (1.0 - config.beta) * t_probs - config.beta * s_probs
    return {"loss": float(np.mean(loss)), "grad_norm": float(np.linalg.norm(grad))}
