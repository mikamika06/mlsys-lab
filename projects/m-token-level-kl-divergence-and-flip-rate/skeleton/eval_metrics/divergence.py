import numpy as np


def compute_kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    """Compute token-level KL divergence KL(P || Q) from unnormalized logits."""
    raise NotImplementedError


def compute_flip_rate(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Compute top-1 token flip rate between baseline and quantized logits."""
    raise NotImplementedError
