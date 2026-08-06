import numpy as np


def compute_attention_kl_divergence(
    student_attn: np.ndarray,
    teacher_attn: np.ndarray,
    eps: float = 1e-12
) -> float:
    """Computes per-layer attention map KL divergence between teacher and student."""
    p = np.clip(teacher_attn, eps, 1.0)
    q = np.clip(student_attn, eps, 1.0)

    p = p / np.sum(p, axis=-1, keepdims=True)
    q = q / np.sum(q, axis=-1, keepdims=True)

    kl = np.sum(p * (np.log(p) - np.log(q)), axis=-1)
    return float(np.mean(kl))
