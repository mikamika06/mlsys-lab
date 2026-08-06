import numpy as np


def compute_attention_kl_divergence(
    student_attn: np.ndarray,
    teacher_attn: np.ndarray,
    eps: float = 1e-12
) -> float:
    """Computes per-layer attention map KL divergence between teacher and student."""
    raise NotImplementedError
