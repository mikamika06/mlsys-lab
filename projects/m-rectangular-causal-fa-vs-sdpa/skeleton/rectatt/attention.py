import numpy as np


def sdpa_rectangular_causal(q: np.ndarray, k: np.ndarray, v: np.ndarray, alignment: str = "bottom_right") -> np.ndarray:
    """Computes scaled dot-product attention with rectangular causal mask."""
    raise NotImplementedError


def flash_attn_sim(q: np.ndarray, k: np.ndarray, v: np.ndarray, is_causal: bool = True, alignment: str = "bottom_right") -> np.ndarray:
    """Simulates FlashAttention causal attention with explicitly aligned bottom-right or top-left causal rules."""
    raise NotImplementedError
