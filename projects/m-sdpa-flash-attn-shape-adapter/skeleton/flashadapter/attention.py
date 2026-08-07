import numpy as np


def flash_attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = False,
    sm_scale: float | None = None,
):
    """Compute reference FlashAttention output [B, N, H, D] and LSE_2 [B, H, N]."""
    raise NotImplementedError
