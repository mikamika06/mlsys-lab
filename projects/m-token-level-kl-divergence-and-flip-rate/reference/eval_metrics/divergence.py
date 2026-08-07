import numpy as np


def compute_kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    """Compute token-level KL divergence KL(P || Q) from unnormalized logits."""
    p_logits = p_logits - np.max(p_logits, axis=-1, keepdims=True)
    q_logits = q_logits - np.max(q_logits, axis=-1, keepdims=True)

    p_exp = np.exp(p_logits)
    p_probs = p_exp / np.sum(p_exp, axis=-1, keepdims=True)
    p_logprobs = p_logits - np.log(np.sum(p_exp, axis=-1, keepdims=True))

    q_exp = np.exp(q_logits)
    q_logprobs = q_logits - np.log(np.sum(q_exp, axis=-1, keepdims=True))

    kl = np.sum(p_probs * (p_logprobs - q_logprobs), axis=-1)
    return np.maximum(kl, 0.0)


def compute_flip_rate(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Compute top-1 token flip rate between baseline and quantized logits."""
    p_top = np.argmax(p_logits, axis=-1)
    q_top = np.argmax(q_logits, axis=-1)
    return float(np.mean(p_top != q_top))
