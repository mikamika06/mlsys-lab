import numpy as np


def sdpa_with_additive_bias(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, bias: np.ndarray, scale: float
) -> np.ndarray:
    """Scaled dot-product attention with an additive bias (padding mask,
    ALiBi, relative position bias, ...).

    q: (n_q, d), k: (n_k, d), v: (n_k, d_v), bias: (n_q, n_k).
    Returns (n_q, d_v).

    BUG: the bias is folded into the pre-scale QK^T product, so it gets
    multiplied by `scale` along with the dot products instead of being
    added afterward, unscaled.
    """
    logits = (q @ k.T + bias) * scale  # BUG: should be (q @ k.T) * scale + bias
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    w = np.exp(logits)
    w = w / np.sum(w, axis=-1, keepdims=True)
    return w @ v
