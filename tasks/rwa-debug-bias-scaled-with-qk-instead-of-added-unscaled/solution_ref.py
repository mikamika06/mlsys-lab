import numpy as np


def sdpa_with_additive_bias(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, bias: np.ndarray, scale: float
) -> np.ndarray:
    """Scaled dot-product attention with an additive bias (padding mask,
    ALiBi, relative position bias, ...).

    Matches the real formula used by e.g. torch.nn.functional's
    scaled_dot_product_attention with a float attn_mask: the QK^T product
    is scaled FIRST, and the bias is added AFTER scaling, unscaled. The
    bias represents a fixed logit offset (e.g. "-1e9 to mask this key" or
    an ALiBi slope*distance term) -- it must not be shrunk by `scale`.

    q: (n_q, d), k: (n_k, d), v: (n_k, d_v), bias: (n_q, n_k).
    Returns (n_q, d_v).
    """
    logits = (q @ k.T) * scale
    logits = logits + bias
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    w = np.exp(logits)
    w = w / np.sum(w, axis=-1, keepdims=True)
    return w @ v
