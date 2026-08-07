import numpy as np


def sdpa_to_flash(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    """Convert [B, H, N, D] SDPA layout to [B, N, H, D] FlashAttention layout."""
    raise NotImplementedError


def flash_to_sdpa(out: np.ndarray):
    """Convert [B, N, H, D] FlashAttention layout to [B, H, N, D] SDPA layout."""
    raise NotImplementedError


def lse_sdpa_to_flash(lse_e: np.ndarray) -> np.ndarray:
    """Convert base-e LSE [B, H, N] to base-2 LSE [B, H, N]."""
    raise NotImplementedError


def lse_flash_to_sdpa(lse_2: np.ndarray) -> np.ndarray:
    """Convert base-2 LSE [B, H, N] to base-e LSE [B, H, N]."""
    raise NotImplementedError
