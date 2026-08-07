import math

def sdpa_with_additive_bias(q: list[list[float]], k: list[list[float]], v: list[list[float]], bias: list[list[float]], scale: float) -> list[list[float]]:
    """Scaled dot-product attention with an additive bias (padding mask,
    ALiBi, relative position bias, ...).

    q: (n_q, d), k: (n_k, d), v: (n_k, d_v), bias: (n_q, n_k).
    Returns (n_q, d_v).

    BUG: the bias is folded into the pre-scale QK^T product, so it gets
    multiplied by `scale` along with the dot products instead of being
    added afterward, unscaled.
    """
    raise NotImplementedError('your code here')
